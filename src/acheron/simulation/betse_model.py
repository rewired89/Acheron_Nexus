"""Bioelectric simulation backend wrapping real BETSE (BioElectric Tissue
Simulation Engine) runs, for Prediction mode's bioelectric leg alongside
`grn_model.py`'s gene-regulatory leg (see `rag/hypothesis_engine.py`'s
`run_prediction_mode`).

WHAT THIS ACTUALLY DOES: shells out to the real `betse` CLI (config -> seed
-> init -> sim, twice: once unperturbed, once with a target ion channel's
conductance scaled down) and parses BETSE's own logged "Final average cell
Vmem" from each run's real output. It never computes or guesses a Vmem
value itself -- every number here is BETSE's own electrodiffusion physics
engine's real result, or this module raises rather than filling a gap.
`pip install betse` (the `simulation` extra) is required; a smoke test in
this project's own development showed BETSE 1.5.0 completes a full
config+seed+init+sim cycle on its default ~225-cell cluster in well under a
minute, which is why this module shells out to a real run rather than
building a second, lower-fidelity bioelectric ODE the way `grn_model.py`
does for gene networks (BETSE already IS the physically-grounded
electrodiffusion engine; there is no reason to reinvent a worse one).

WHAT THIS IS NOT: a gene-aware simulator. BETSE models membrane biophysics
by ion-channel CLASS (Na/K/Ca/Cl, with named kinetic types like Nav1p3,
Kv1p5) -- it has no concept of "yugO" or any other gene. Bridging a Phase 3
gene perturbation to a BETSE channel-class perturbation therefore requires
an explicit mapping, which this module will only make from literal cited
text (an evidence_text string containing "potassium"/"k+"/etc., not a
guess) or from an explicit caller-supplied `channel_class` -- if neither is
available, the bioelectric leg still runs as an honest UNPERTURBED baseline
(so a real Vmem number is still returned), with a warning that no channel
mapping was found rather than a silently invented one.

SCOPE LIMIT: only Na and K have a pre-populated dynamic channel entry
(`Nav`/`Kv`+`K_Leak`) in BETSE's own default config template. Ca and Cl
would require hand-authoring a custom channel block with biophysical
parameters (gating kinetics, half-activation voltage, etc.) this module
has no cited source for -- rather than invent one, `simulate_bioelectric`
raises `ChannelClassUnsupported` for those two classes.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

DEFAULT_MAX_DM_FRACTION = 0.1
DEFAULT_TIMEOUT_SECONDS = 240

# Only Na and K have a dynamic channel entry in BETSE's own default
# `betse config` template (section `general network: channels:`) -- see
# module docstring's SCOPE LIMIT.
_SUPPORTED_CHANNEL_CLASSES = ("Na", "K")

# Keyword -> BETSE channel class, matched literally against a record's own
# cited evidence_text (never inferred from the gene name alone).
_CHANNEL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Na": ("sodium", "na+"),
    "K": ("potassium", "k+"),
    "Ca": ("calcium", "ca2+"),
    "Cl": ("chloride", "cl-"),
}

_VMEM_RE = re.compile(r"Final average cell Vmem:\s*(-?[\d.]+)\s*mV")
_CELL_COUNT_RE = re.compile(r"This world contains (\d+) cells")


class BETSEBackendUnavailable(RuntimeError):
    """Raised when the `betse` CLI isn't installed/runnable."""

    def __init__(self, message: Optional[str] = None) -> None:
        super().__init__(
            message
            or "Bioelectric simulation requires BETSE. Install with: pip install betse "
            "(or: pip install -e \".[simulation]\")"
        )


class ChannelClassUnsupported(ValueError):
    """Raised for a channel class this module has no default BETSE entry for."""


class BioelectricSimulationResult(BaseModel):
    channel_class: Optional[str]
    channel_class_cited: bool
    max_dm_fraction: float
    baseline_vmem_mV: float
    perturbed_vmem_mV: float
    delta_vmem_mV: float
    cell_count: Optional[int] = None
    runtime_seconds: float
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def infer_channel_class(evidence_texts: list[str]) -> tuple[Optional[str], bool]:
    """Scan literal cited evidence text for an ion-channel-class keyword.

    Returns (channel_class, cited). Never guesses from the gene name alone --
    only a literal keyword match in text a real source actually wrote counts.
    """
    joined = " ".join(t for t in evidence_texts if t).lower()
    for channel_class, keywords in _CHANNEL_KEYWORDS.items():
        for keyword in keywords:
            if keyword in joined:
                return channel_class, True
    return None, False


def _check_betse_available() -> None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "betse", "--version"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as exc:
        raise BETSEBackendUnavailable() from exc
    if result.returncode != 0:
        raise BETSEBackendUnavailable(
            f"BETSE is installed but failed to run:\n{result.stderr[-2000:]}"
        )


def _run_betse(args: list[str], timeout: int) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "betse", "--headless", *args],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"BETSE {' '.join(args)} failed (exit {proc.returncode}):\n"
            f"{proc.stdout[-3000:]}\n{proc.stderr[-2000:]}"
        )
    return proc.stdout


def _scale_channel_class(config_path: Path, channel_class: str, max_dm_fraction: float) -> None:
    """Reduce `max Dm` for every dynamic channel entry of `channel_class` in
    the `general network: channels:` list. Uses ruamel.yaml (already a BETSE
    dependency) in round-trip mode to avoid corrupting this config's many
    inline comments and BETSE-specific scalar conventions (e.g. bare `None`)."""
    from ruamel.yaml import YAML

    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml_rt.load(fh)

    channels = ((config.get("general network") or {}).get("channels")) or []
    matched = 0
    for entry in channels:
        if entry.get("channel class") == channel_class:
            entry["max Dm"] = float(entry["max Dm"]) * max_dm_fraction
            matched += 1

    if matched == 0:
        raise ChannelClassUnsupported(
            f"No default BETSE channel entry found for class '{channel_class}' "
            f"in this config's 'general network: channels:' list."
        )

    with config_path.open("w", encoding="utf-8") as fh:
        yaml_rt.dump(config, fh)


def _generate_config(work_dir: Path, label: str) -> Path:
    """Runs `betse config` for one variant ('baseline' or 'perturbed'),
    each in its own subdirectory so the two runs never share state."""
    config_path = work_dir / label / f"{label}.yaml"
    _run_betse(["config", str(config_path)], timeout=60)
    return config_path


def simulate_bioelectric(
    channel_class: Optional[str] = None,
    *,
    max_dm_fraction: float = DEFAULT_MAX_DM_FRACTION,
    evidence_texts: Optional[list[str]] = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> BioelectricSimulationResult:
    """Run a real BETSE baseline simulation and, if a channel class is known
    (explicit or inferred from cited evidence_texts), a second run with that
    channel class's conductance scaled by `max_dm_fraction` (0=fully
    blocked, 1=unchanged -- same convention as grn_model.py's
    knockdown_fraction). Returns both runs' real logged Vmem and the delta.

    Raises BETSEBackendUnavailable if BETSE isn't installed, or
    ChannelClassUnsupported if channel_class isn't Na/K (see module
    docstring's SCOPE LIMIT).
    """
    import time

    if not (0.0 <= max_dm_fraction <= 1.0):
        raise ValueError("max_dm_fraction must be between 0.0 and 1.0")

    _check_betse_available()

    warnings: list[str] = []
    channel_class_cited = False
    if channel_class is not None:
        channel_class_cited = True  # explicitly supplied by the caller
    elif evidence_texts:
        channel_class, channel_class_cited = infer_channel_class(evidence_texts)
        if channel_class is None:
            warnings.append(
                "No ion-channel-class keyword (sodium/potassium/calcium/chloride) "
                "found in the cited evidence text -- running an unperturbed BETSE "
                "baseline only. Pass channel_class explicitly to target a specific "
                "channel."
            )
    else:
        warnings.append(
            "No channel_class provided and no evidence_texts to infer one from -- "
            "running an unperturbed BETSE baseline only."
        )

    if channel_class is not None and channel_class not in _SUPPORTED_CHANNEL_CLASSES:
        raise ChannelClassUnsupported(
            f"channel_class '{channel_class}' has no default BETSE channel entry "
            f"(only {_SUPPORTED_CHANNEL_CLASSES} are supported -- see module "
            f"docstring's SCOPE LIMIT for Ca/Cl)."
        )

    start = time.monotonic()
    root = Path(tempfile.mkdtemp(prefix="acheron_betse_"))
    try:
        # Baseline (unperturbed) run.
        baseline_cfg = _generate_config(root, "baseline")
        _run_betse(["seed", str(baseline_cfg)], timeout_seconds)
        _run_betse(["init", str(baseline_cfg)], timeout_seconds)
        baseline_sim_log = _run_betse(["sim", str(baseline_cfg)], timeout_seconds)
        baseline_vmem = _extract_vmem(baseline_sim_log)
        cell_count = _extract_cell_count(baseline_sim_log)

        if channel_class is None:
            perturbed_vmem = baseline_vmem
        else:
            perturbed_cfg = _generate_config(root, "perturbed")
            _scale_channel_class(perturbed_cfg, channel_class, max_dm_fraction)
            _run_betse(["seed", str(perturbed_cfg)], timeout_seconds)
            _run_betse(["init", str(perturbed_cfg)], timeout_seconds)
            perturbed_sim_log = _run_betse(["sim", str(perturbed_cfg)], timeout_seconds)
            perturbed_vmem = _extract_vmem(perturbed_sim_log)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    runtime = time.monotonic() - start

    notes = [
        "Both Vmem values are BETSE's own real electrodiffusion-engine output "
        "('Final average cell Vmem' from its own log), not computed by this "
        "module -- see the module docstring for what this does and does not do.",
    ]
    if channel_class is not None:
        notes.append(
            f"Perturbed run scaled every '{channel_class}'-class dynamic channel's "
            f"max Dm (membrane diffusion constant) by {max_dm_fraction:.0%} of its "
            f"default, representing a sustained {100 * (1 - max_dm_fraction):.0f}% "
            f"reduction in that channel class's conductance."
        )

    return BioelectricSimulationResult(
        channel_class=channel_class,
        channel_class_cited=channel_class_cited,
        max_dm_fraction=max_dm_fraction,
        baseline_vmem_mV=baseline_vmem,
        perturbed_vmem_mV=perturbed_vmem,
        delta_vmem_mV=perturbed_vmem - baseline_vmem,
        cell_count=cell_count,
        runtime_seconds=runtime,
        warnings=warnings,
        notes=notes,
    )


def _extract_vmem(log_text: str) -> float:
    match = _VMEM_RE.search(log_text)
    if not match:
        raise RuntimeError(
            "Could not find 'Final average cell Vmem' in BETSE's output -- "
            "refusing to fabricate a value. Tail of output:\n" + log_text[-2000:]
        )
    return float(match.group(1))


def _extract_cell_count(log_text: str) -> Optional[int]:
    match = _CELL_COUNT_RE.search(log_text)
    return int(match.group(1)) if match else None
