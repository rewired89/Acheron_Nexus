"""Gene-regulatory-network (GRN) simulation backend, built on Tellurium's
underlying engine (libRoadRunner + Antimony), consuming Phase 3's structured
parameter store (`extraction/parameter_extractor.py`'s `ParameterRecord`s).

WHAT THIS IS: given an organism and a gene to perturb (knock down), this
module builds a small directed network from whatever `(subject)
-[relationship_type]-> (object)` facts Phase 3 has actually extracted for
that organism, converts it to an SBML model via Antimony, and simulates how
the perturbation propagates downstream over time using libRoadRunner (the
same ODE-integration engine Tellurium wraps -- this module talks to it
directly rather than importing the much heavier `tellurium` package, which
bundles plotting/Jupyter/analysis tooling this CLI-driven backend does not
use; `pip install antimony libroadrunner` is enough).

WHAT THIS IS NOT: a validated kinetic model. Per this project's own
no-invented-numbers rule (see CLAUDE.md), almost every SubtiWiki/PlanMine
`ParameterRecord.rate_or_affinity` is the literal string "UNKNOWN" --
`extraction/parameter_extractor.py` only ever fills that field with a value
a source explicitly stated. This module never invents a missing rate: any
edge without a cited rate constant is simulated with an explicitly-labeled
placeholder (`DEFAULT_RATE_CONSTANT`) and reported back in
`GRNSimulationResult.unknown_parameters`, never silently. The resulting
`confidence_score` is exactly the fraction of the simulated network's edges
that carried a cited rate constant -- a network built entirely from UNKNOWN
edges reports confidence 0.0, not a plausible-looking number.

MODELING CHOICES (structural, not biological facts -- see `notes` on every
result):
  - Each gene is a deviation-from-baseline species (baseline = 1.0). A gene
    with no perturbation and no active regulatory input stays at baseline
    (dX/dt = decay*(1-X) + sum of incoming edge terms). This is a standard
    linearized GRN convention, the same category of engineering
    simplification as `gardner_toggle_circuit.py`'s dimensionless Gardner
    ODE elsewhere in this project's sister repo (Acheron) -- not a
    citation-backed rate law.
  - `decay_rate` is a single structural constant applied to every gene, not
    a per-gene cited value, and is therefore never counted toward
    `confidence_score` (only per-edge rate constants are).
  - The perturbed gene is held at a fixed (`const`) concentration
    (`knockdown_fraction` x baseline) for the entire run, modeling a
    sustained knockdown/knockout line, not a transient pulse.
  - "regulated_by" edges run causally object->subject (the regulator drives
    the regulated gene); "activates"/"inhibits"/"regulates" edges run
    subject->object; "interacts_with"/"binds" (physical, not confirmed
    transcriptional) edges are modeled as a symmetric coupling in both
    directions -- flagged `relationship_class="physical_interaction"` in
    every such edge so this simplification is visible in the output, not
    just this docstring.
  - Activation vs. repression sign: "activates"/"inhibits" relationship
    types state it directly. Everything else (SubtiWiki's "regulated_by",
    "regulates", "interacts_with", literature "binds") has no structured
    sign field, so this module searches the record's own `evidence_text`
    for an explicit positive/negative cue (e.g. SubtiWiki's own
    "(..., negative)" mode annotation) before falling back to an assumed
    default of "activation" -- and every assumed sign is flagged in
    `unknown_parameters` exactly like an assumed rate constant.
  - SubtiWiki's own "regulates" records are a *regulon summary*
    ("regulates 5 gene(s) and 2 operon(s)"), not a specific target gene --
    those carry no usable object identity and are excluded from the
    network entirely, never guessed at.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from acheron.extraction.parameter_extractor import (
    ConfidenceTier,
    ParameterRecord,
    ParameterStore,
)

# ======================================================================
# Structural modeling constants (engineering defaults, not cited values --
# see module docstring). None of these are ever reported as "cited".
# ======================================================================
BASELINE_EXPRESSION = 1.0
DEFAULT_RATE_CONSTANT = 1.0
DEFAULT_DECAY_RATE = 0.1
DEFAULT_KNOCKDOWN_FRACTION = 0.1
DEFAULT_DURATION = 100.0
DEFAULT_N_POINTS = 101
DEFAULT_MAX_HOPS = 2


class GRNBackendUnavailable(RuntimeError):
    """Raised when antimony/libroadrunner aren't installed."""

    def __init__(self, message: Optional[str] = None) -> None:
        super().__init__(
            message
            or "GRN simulation requires the 'antimony' and 'libroadrunner' packages. "
            "Install with: pip install antimony libroadrunner "
            "(or: pip install -e \".[simulation]\")"
        )


# ======================================================================
# Organism-name resolution -- mirrors parameter_extractor.py's
# _ORGANISM_TARGET_MAP idea, inverted: free-text CLI input -> the exact
# organism string collectors actually write onto Paper/TextChunk records.
# ======================================================================
_ORGANISM_ALIASES = {
    "bacteria": "B. subtilis",
    "b. subtilis": "B. subtilis",
    "b subtilis": "B. subtilis",
    "bacillus": "B. subtilis",
    "bacillus subtilis": "B. subtilis",
    "subtilis": "B. subtilis",
    "planarian": "S. mediterranea",
    "s. mediterranea": "S. mediterranea",
    "schmidtea": "S. mediterranea",
    "schmidtea mediterranea": "S. mediterranea",
}


def _resolve_organism(organism: str) -> str:
    return _ORGANISM_ALIASES.get(organism.strip().lower(), organism.strip())


def _tier_for_fraction(fraction: float) -> ConfidenceTier:
    """Same cut points as parameter_extractor._tier_from_score, applied to a
    different quantity (fraction of edges with a cited rate constant) --
    kept as its own function since the two scores measure different things."""
    if fraction >= 0.7:
        return ConfidenceTier.HIGH
    if fraction >= 0.4:
        return ConfidenceTier.MEDIUM
    return ConfidenceTier.LOW


# ======================================================================
# Output shape (kept local to this module, same convention
# parameter_extractor.py uses for ParameterRecord -- nothing added to
# models.py for a Phase-3-consuming module).
# ======================================================================
class GRNEdge(BaseModel):
    source_gene: str
    target_gene: str
    directed: bool
    relationship_type: str
    relationship_class: str  # "regulatory" | "physical_interaction"
    sign: str  # "activation" | "repression"
    sign_cited: bool
    rate_constant: float
    rate_cited: bool
    confidence_score: float
    evidence_text: str
    source: str
    paper_id: str
    record_id: str


class GRNSimulationResult(BaseModel):
    organism: str
    perturbation_gene: str
    knockdown_fraction: float
    genes: list[str]
    edges: list[GRNEdge]
    n_cited_edges: int
    n_total_edges: int
    confidence_score: float
    confidence_tier: ConfidenceTier
    timepoints: list[float]
    trajectories: dict[str, list[float]]
    unknown_parameters: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ======================================================================
# Record -> network edge
# ======================================================================
_REGULATORY_TYPES = {"regulates", "regulated_by", "activates", "inhibits"}
_PHYSICAL_TYPES = {"interacts_with", "binds"}
_REGULON_SUMMARY_MARKER = "regulon summary, not individual targets"

_NEGATIVE_KEYWORDS = ("negative", "repress", "inhibit")
_POSITIVE_KEYWORDS = ("positive", "activat", "induc")

_NUMBER_RE = re.compile(r"-?\d+\.?\d*")


def _infer_sign(relationship_type: str, evidence_text: str) -> tuple[str, bool]:
    """Returns (sign, cited). 'cited' is True only when the sign came from
    the relationship_type itself (activates/inhibits) or was found as a
    literal keyword in the record's own evidence_text -- never guessed."""
    if relationship_type == "activates":
        return "activation", True
    if relationship_type == "inhibits":
        return "repression", True
    text = (evidence_text or "").lower()
    if any(keyword in text for keyword in _NEGATIVE_KEYWORDS):
        return "repression", True
    if any(keyword in text for keyword in _POSITIVE_KEYWORDS):
        return "activation", True
    return "activation", False


def _resolve_rate_constant(rate_or_affinity: str) -> tuple[float, bool]:
    """Returns (rate_constant, cited). Only ever uses a literal number found
    in the source's own rate_or_affinity text; UNKNOWN -> the labeled
    placeholder, never a guess."""
    if rate_or_affinity and rate_or_affinity != "UNKNOWN":
        match = _NUMBER_RE.search(rate_or_affinity)
        if match:
            try:
                return float(match.group(0)), True
            except ValueError:
                pass
    return DEFAULT_RATE_CONSTANT, False


def _edge_from_record(record: ParameterRecord) -> Optional[dict]:
    """Converts one ParameterRecord into a network edge dict, or None if the
    record's relationship_type isn't a GRN-usable edge (e.g. PlanMine's
    homology/domain/RNAi-expression annotations, or SubtiWiki's regulon-
    summary "regulates" record, which names no specific target gene)."""
    rtype = record.relationship_type
    if rtype in _REGULATORY_TYPES:
        relationship_class = "regulatory"
    elif rtype in _PHYSICAL_TYPES:
        relationship_class = "physical_interaction"
    else:
        return None

    subject, obj = record.subject.strip(), record.object.strip()
    if not subject or not obj:
        return None
    if rtype == "regulates" and _REGULON_SUMMARY_MARKER in obj:
        return None

    if rtype == "regulated_by":
        source_gene, target_gene, directed = obj, subject, True
    elif rtype in ("regulates", "activates", "inhibits"):
        source_gene, target_gene, directed = subject, obj, True
    else:  # interacts_with / binds
        source_gene, target_gene, directed = subject, obj, False

    sign, sign_cited = _infer_sign(rtype, record.evidence_text)
    rate_constant, rate_cited = _resolve_rate_constant(record.rate_or_affinity)

    return {
        "source_gene": source_gene,
        "target_gene": target_gene,
        "directed": directed,
        "relationship_type": rtype,
        "relationship_class": relationship_class,
        "sign": sign,
        "sign_cited": sign_cited,
        "rate_constant": rate_constant,
        "rate_cited": rate_cited,
        "confidence_score": record.confidence_score,
        "evidence_text": record.evidence_text,
        "source": record.source,
        "paper_id": record.paper_id,
        "record_id": record.record_id,
    }


def _normalize_id(name: str) -> str:
    return name.strip().lower()


def _dedup_key(edge: dict) -> tuple[str, str, str]:
    """Two ParameterRecords (e.g. the same SubtiWiki fact re-extracted from
    two overlapping chunks) can describe the same edge -- dedup so the ODE
    never double-counts one causal relationship as two summed terms."""
    a, b = _normalize_id(edge["source_gene"]), _normalize_id(edge["target_gene"])
    if not edge["directed"]:
        a, b = sorted([a, b])
    return (a, b, edge["relationship_type"])


def _dedup_edges(edges: list[dict]) -> list[dict]:
    best: dict[tuple, dict] = {}
    for edge in edges:
        key = _dedup_key(edge)
        current = best.get(key)
        if current is None:
            best[key] = edge
            continue
        if edge["rate_cited"] and not current["rate_cited"]:
            best[key] = edge
        elif edge["rate_cited"] == current["rate_cited"] and edge["confidence_score"] > current["confidence_score"]:
            best[key] = edge
    return list(best.values())


def _build_downstream_network(
    edges: list[dict], start_norm: str, max_hops: int
) -> tuple[set[str], list[dict]]:
    """BFS strictly along causal direction (source->target) for regulatory
    edges, either direction for physical-interaction edges (no defined
    causal direction to respect), to find genes reachable from the
    perturbation within max_hops. Returns (visited genes, induced subgraph
    of every deduped edge whose endpoints are both reachable)."""
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        s, t = _normalize_id(edge["source_gene"]), _normalize_id(edge["target_gene"])
        adjacency[s].append(t)
        if not edge["directed"]:
            adjacency[t].append(s)

    visited = {start_norm}
    frontier = {start_norm}
    for _ in range(max_hops):
        next_frontier: set[str] = set()
        for gene in frontier:
            for neighbor in adjacency.get(gene, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        if not next_frontier:
            break
        frontier = next_frontier

    included = [
        edge for edge in edges
        if _normalize_id(edge["source_gene"]) in visited and _normalize_id(edge["target_gene"]) in visited
    ]
    return visited, included


# ======================================================================
# Antimony/SBML model construction
# ======================================================================
def _sanitize_sbml_id(name: str, used: set[str]) -> str:
    base = re.sub(r"[^0-9A-Za-z_]", "_", name.strip()) or "gene"
    if base[0].isdigit():
        base = f"g_{base}"
    candidate = base
    i = 2
    while candidate in used:
        candidate = f"{base}_{i}"
        i += 1
    used.add(candidate)
    return candidate


def _build_antimony_model(
    display_by_norm: dict[str, str],
    perturbation_norm: str,
    knockdown_fraction: float,
    edges: list[dict],
    decay_rate: float,
) -> tuple[str, dict[str, str]]:
    used_ids: set[str] = set()
    sbml_id_by_norm: dict[str, str] = {}
    species_lines = []
    for norm, display in display_by_norm.items():
        sid = _sanitize_sbml_id(display, used_ids)
        sbml_id_by_norm[norm] = sid
        if norm == perturbation_norm:
            species_lines.append(f"  species {sid} = {BASELINE_EXPRESSION * knockdown_fraction}")
            species_lines.append(f"  const {sid}")
        else:
            species_lines.append(f"  species {sid} = {BASELINE_EXPRESSION}")

    terms_by_target: dict[str, list[str]] = defaultdict(list)
    param_lines = []
    counter = 0
    for edge in edges:
        s_norm, t_norm = _normalize_id(edge["source_gene"]), _normalize_id(edge["target_gene"])
        sign_mult = "1" if edge["sign"] == "activation" else "-1"

        counter += 1
        pname = f"k_{counter}"
        param_lines.append(f"  {pname} = {edge['rate_constant']}")
        terms_by_target[t_norm].append(
            f"{sign_mult}*{pname}*({sbml_id_by_norm[s_norm]} - {BASELINE_EXPRESSION})"
        )

        if not edge["directed"]:
            counter += 1
            pname2 = f"k_{counter}"
            param_lines.append(f"  {pname2} = {edge['rate_constant']}")
            terms_by_target[s_norm].append(
                f"{sign_mult}*{pname2}*({sbml_id_by_norm[t_norm]} - {BASELINE_EXPRESSION})"
            )

    ode_lines = []
    for norm, display in display_by_norm.items():
        if norm == perturbation_norm:
            continue
        sid = sbml_id_by_norm[norm]
        rhs = f"decay*({BASELINE_EXPRESSION} - {sid})"
        terms = terms_by_target.get(norm, [])
        if terms:
            rhs += " + " + " + ".join(terms)
        ode_lines.append(f"  {sid}' = {rhs}")

    lines = ["model grn_perturbation"]
    lines.extend(species_lines)
    lines.append(f"  decay = {decay_rate}")
    lines.extend(param_lines)
    lines.extend(ode_lines)
    lines.append("end")
    return "\n".join(lines), sbml_id_by_norm


# ======================================================================
# Public entry point
# ======================================================================
def simulate_grn(
    organism: str,
    perturbation_gene: str,
    *,
    knockdown_fraction: float = DEFAULT_KNOCKDOWN_FRACTION,
    duration: float = DEFAULT_DURATION,
    n_points: int = DEFAULT_N_POINTS,
    max_hops: int = DEFAULT_MAX_HOPS,
    decay_rate: float = DEFAULT_DECAY_RATE,
    params_dir: Optional[Path] = None,
) -> GRNSimulationResult:
    """Simulate the downstream effect of knocking down `perturbation_gene`
    in `organism`, using only Phase 3's cited ParameterRecords.

    Raises ValueError if the organism has no extracted parameter records at
    all (run `acheron index` then extract_from_store first), or
    GRNBackendUnavailable if antimony/libroadrunner aren't installed.
    """
    if not (0.0 <= knockdown_fraction <= 1.0):
        raise ValueError("knockdown_fraction must be between 0.0 and 1.0")

    resolved_organism = _resolve_organism(organism)
    store = ParameterStore(params_dir)
    records = store.list_records(organism=resolved_organism)

    if not records:
        available = sorted({r.organism for r in store.list_records() if r.organism})
        raise ValueError(
            f"No parameter records found for organism '{organism}' (resolved to "
            f"'{resolved_organism}'). Run 'acheron index' and extract parameters "
            f"first. Organisms currently in the parameter store: "
            f"{available or '(none extracted yet)'}"
        )

    warnings: list[str] = []
    edges_all = [e for e in (_edge_from_record(r) for r in records) if e is not None]
    deduped_edges = _dedup_edges(edges_all)

    display_by_norm: dict[str, str] = {}
    for edge in edges_all:
        display_by_norm.setdefault(_normalize_id(edge["source_gene"]), edge["source_gene"])
        display_by_norm.setdefault(_normalize_id(edge["target_gene"]), edge["target_gene"])

    perturbation_norm = _normalize_id(perturbation_gene)
    if perturbation_norm not in display_by_norm:
        display_by_norm[perturbation_norm] = perturbation_gene
        warnings.append(
            f"'{perturbation_gene}' does not appear as a subject or object in any "
            f"extracted parameter record for {resolved_organism} -- simulating it "
            f"as an isolated node with decay-only dynamics. This reflects a gap in "
            f"the indexed evidence, not a biological claim about {perturbation_gene}."
        )

    visited, included_edges = _build_downstream_network(deduped_edges, perturbation_norm, max_hops)
    display_by_norm = {norm: display for norm, display in display_by_norm.items() if norm in visited}

    if not included_edges:
        warnings.append(
            f"No regulatory or physical-interaction edges connect '{perturbation_gene}' "
            f"to any other gene within {max_hops} hop(s) in the current parameter "
            f"store. Showing isolated decay dynamics for '{perturbation_gene}' only."
        )

    n_cited = sum(1 for e in included_edges if e["rate_cited"])
    n_total = len(included_edges)
    confidence_score = (n_cited / n_total) if n_total else 0.0
    confidence_tier = _tier_for_fraction(confidence_score)

    unknown_parameters = [
        f"{e['source_gene']} -[{e['relationship_type']}]-> {e['target_gene']}: no cited "
        f"rate/affinity value, used placeholder rate_constant={DEFAULT_RATE_CONSTANT} "
        f"(record {e['record_id']}, paper {e['paper_id']})"
        for e in included_edges if not e["rate_cited"]
    ]
    unknown_parameters.extend(
        f"{e['source_gene']} -[{e['relationship_type']}]-> {e['target_gene']}: "
        f"activation/repression sign not stated in the cited evidence text, "
        f"defaulted to 'activation' (record {e['record_id']})"
        for e in included_edges if not e["sign_cited"]
    )

    try:
        import antimony
        import roadrunner
    except ImportError as exc:
        raise GRNBackendUnavailable() from exc

    antimony_src, sbml_id_by_norm = _build_antimony_model(
        display_by_norm, perturbation_norm, knockdown_fraction, included_edges, decay_rate
    )

    load_code = antimony.loadAntimonyString(antimony_src)
    if load_code < 0:
        raise RuntimeError(f"Failed to build GRN model: {antimony.getLastError()}")
    sbml = antimony.getSBMLString()
    antimony.clearPreviousLoads()

    runner = roadrunner.RoadRunner(sbml)
    sim = runner.simulate(0, duration, n_points)

    colnames = list(sim.colnames)
    norm_by_sbml_id = {sid: norm for norm, sid in sbml_id_by_norm.items()}
    timepoints = [float(row[colnames.index("time")]) for row in sim]
    trajectories: dict[str, list[float]] = {}
    for col_idx, col_name in enumerate(colnames):
        if col_name == "time":
            continue
        sid = col_name.strip("[]")
        norm = norm_by_sbml_id.get(sid)
        display = display_by_norm.get(norm, sid)
        trajectories[display] = [float(row[col_idx]) for row in sim]

    perturbed_display = display_by_norm[perturbation_norm]
    trajectories[perturbed_display] = [BASELINE_EXPRESSION * knockdown_fraction] * len(timepoints)

    notes = [
        "Model is a linearized deviation-from-baseline ODE: dX/dt = decay*(1-X) + "
        "sum(sign*k*(driver-1)) per incoming edge -- a structural modeling "
        "convention, not a cited biological rate law. decay_rate and the "
        "baseline=1.0 set point are not counted in confidence_score below "
        "(only per-edge rate constants are).",
        "Cited rate_or_affinity values are used as each edge's rate constant "
        "VERBATIM, with no unit harmonization across sources (Kd/Ki/kcat/EC50/"
        "IC50 are not interchangeable) -- treat relative magnitudes as "
        "illustrative, not physiologically precise.",
        "'physical_interaction' edges (interacts_with/binds) are modeled as a "
        "symmetric coupling, not confirmed transcriptional regulation -- a "
        "protein-protein interaction does not by itself establish which gene's "
        "expression drives which other gene's expression.",
        f"Knockdown modeled as a fixed (const) species at "
        f"{knockdown_fraction:.0%} of baseline for the full simulated duration "
        f"(sustained knockdown/knockout line, not a transient perturbation).",
    ]

    return GRNSimulationResult(
        organism=resolved_organism,
        perturbation_gene=perturbation_gene,
        knockdown_fraction=knockdown_fraction,
        genes=sorted(display_by_norm.values()),
        edges=[GRNEdge(**e) for e in included_edges],
        n_cited_edges=n_cited,
        n_total_edges=n_total,
        confidence_score=confidence_score,
        confidence_tier=confidence_tier,
        timepoints=timepoints,
        trajectories=trajectories,
        unknown_parameters=unknown_parameters,
        warnings=warnings,
        notes=notes,
    )
