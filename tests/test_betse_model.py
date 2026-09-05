"""Tests for the BETSE bioelectric wrapper (src/acheron/simulation/betse_model.py).

Fast, offline tests for the pure logic (channel-class inference, YAML
scaling, output parsing). The real end-to-end BETSE run (config -> seed ->
init -> sim, twice) takes about a minute and is exercised separately, not
on every test run -- see `test_simulate_bioelectric_real_run_smoke`, gated
behind an explicit opt-in env var so the normal test suite stays fast and
offline, matching this project's own convention for the VectorStore's slow
first-run test.
"""

import os
import textwrap

import pytest

from acheron.simulation import betse_model as bm


# ======================================================================
# Channel-class inference -- never guesses from a gene name alone
# ======================================================================
def test_infer_channel_class_potassium_keyword():
    assert bm.infer_channel_class(["YugO K+ efflux activates KinC"]) == ("K", True)


def test_infer_channel_class_sodium_keyword():
    assert bm.infer_channel_class(["Nav channel conducts sodium ions"]) == ("Na", True)


def test_infer_channel_class_no_keyword_found():
    assert bm.infer_channel_class(["Regulated by SinR"]) == (None, False)


def test_infer_channel_class_empty_list():
    assert bm.infer_channel_class([]) == (None, False)


# ======================================================================
# Vmem / cell-count log parsing -- never fabricates on a parse failure
# ======================================================================
def test_extract_vmem_parses_real_betse_log_line():
    log = "some other output\nFinal average cell Vmem: -43.084 mV\nmore output"
    assert bm._extract_vmem(log) == -43.084


def test_extract_vmem_raises_rather_than_fabricating():
    with pytest.raises(RuntimeError, match="refusing to fabricate"):
        bm._extract_vmem("no vmem line here at all")


def test_extract_cell_count_parses_real_betse_log_line():
    log = "[__main__.py] This world contains 225 cells."
    assert bm._extract_cell_count(log) == 225


def test_extract_cell_count_missing_returns_none():
    assert bm._extract_cell_count("no cell count here") is None


# ======================================================================
# Config YAML scaling -- operates on a minimal fixture shaped like BETSE's
# own 'general network: channels:' section, not a live BETSE install.
# ======================================================================
_FIXTURE_CONFIG = textwrap.dedent("""\
    general network:
      implement network: true
      channels:
      - name: Nav
        channel class: Na
        channel type: Nav1p3
        max Dm: 2.0e-14
        apply to: all
        init active: false
      - name: Kv
        channel class: K
        channel type: Kv1p5
        max Dm: 1.0e-15
        apply to: all
        init active: false
      - name: K_Leak
        channel class: K
        channel type: KLeak
        max Dm: 0.6e-17
        apply to: all
        init active: true
    """)


def test_scale_channel_class_scales_all_matching_entries(tmp_path):
    config_path = tmp_path / "sim.yaml"
    config_path.write_text(_FIXTURE_CONFIG, encoding="utf-8")

    bm._scale_channel_class(config_path, "K", 0.1)

    from ruamel.yaml import YAML
    yaml_rt = YAML()
    with config_path.open() as fh:
        config = yaml_rt.load(fh)

    channels = {c["name"]: c for c in config["general network"]["channels"]}
    assert channels["Kv"]["max Dm"] == pytest.approx(1.0e-15 * 0.1)
    assert channels["K_Leak"]["max Dm"] == pytest.approx(0.6e-17 * 0.1)
    # Na entry must be untouched.
    assert channels["Nav"]["max Dm"] == pytest.approx(2.0e-14)


def test_scale_channel_class_raises_for_absent_class(tmp_path):
    config_path = tmp_path / "sim.yaml"
    config_path.write_text(_FIXTURE_CONFIG, encoding="utf-8")

    with pytest.raises(bm.ChannelClassUnsupported):
        bm._scale_channel_class(config_path, "Cl", 0.1)


# ======================================================================
# simulate_bioelectric argument validation and unsupported-class guard
# (fast -- fails before any subprocess is spawned)
# ======================================================================
def test_simulate_bioelectric_rejects_out_of_range_fraction():
    with pytest.raises(ValueError, match="max_dm_fraction"):
        bm.simulate_bioelectric(channel_class="K", max_dm_fraction=1.5)


def test_simulate_bioelectric_rejects_unsupported_channel_class():
    with pytest.raises(bm.ChannelClassUnsupported):
        bm.simulate_bioelectric(channel_class="Ca")


# ======================================================================
# Real end-to-end BETSE run -- slow (~1 minute), opt-in only.
# Set ACHERON_RUN_SLOW_BETSE_TEST=1 to exercise it (requires
# `pip install betse`; validated manually during this module's development
# to complete a full config+seed+init+sim cycle on the default ~225-cell
# cluster in well under a minute).
# ======================================================================
@pytest.mark.skipif(
    not os.environ.get("ACHERON_RUN_SLOW_BETSE_TEST"),
    reason="Real BETSE run takes ~1-2 minutes; set ACHERON_RUN_SLOW_BETSE_TEST=1 to run it.",
)
def test_simulate_bioelectric_real_run_smoke():
    result = bm.simulate_bioelectric(channel_class="K", max_dm_fraction=0.1)
    assert result.channel_class == "K"
    assert result.channel_class_cited is True
    assert result.cell_count and result.cell_count > 0
    # Reducing K+ conductance should depolarize the cell (Vmem becomes less
    # negative), the physically correct direction for this perturbation.
    assert result.perturbed_vmem_mV > result.baseline_vmem_mV
