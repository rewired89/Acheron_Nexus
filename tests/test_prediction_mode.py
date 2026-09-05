"""Tests for MODE 6 (prediction) in rag/hypothesis_engine.py: the combined
GRN + bioelectric prediction, its derived confidence score, the cited-vs-
assumed split, and the experiment_designer.py feed.

Fast and offline: simulate_grn/simulate_bioelectric are monkeypatched with
small fixture results rather than running real libRoadRunner/BETSE
processes on every test -- those are exercised for real in
test_grn_model.py's end-to-end test and test_betse_model.py's opt-in slow
test respectively. This file tests only run_prediction_mode's own
combination logic.
"""

import pytest

from acheron.models import NexusMode
from acheron.rag import hypothesis_engine as he
from acheron.simulation.betse_model import BETSEBackendUnavailable, BioelectricSimulationResult
from acheron.simulation.grn_model import ConfidenceTier, GRNEdge, GRNSimulationResult


def _grn_result(n_cited: int, n_total: int, organism="B. subtilis", perturbation="yugO"):
    edges = []
    for i in range(n_total):
        edges.append(GRNEdge(
            source_gene=perturbation, target_gene=f"gene{i}", directed=True,
            relationship_type="activates", relationship_class="regulatory",
            sign="activation", sign_cited=True,
            rate_constant=5.0 if i < n_cited else 1.0, rate_cited=i < n_cited,
            confidence_score=0.5, evidence_text=f"{perturbation} activates gene{i}",
            source="subtiwiki", paper_id="subtiwiki:gene:x", record_id=f"param-{i}",
        ))
    unknown = [
        f"{perturbation} -[activates]-> gene{i}: no cited rate/affinity value"
        for i in range(n_cited, n_total)
    ]
    confidence = (n_cited / n_total) if n_total else 0.0
    tier = ConfidenceTier.HIGH if confidence >= 0.7 else ConfidenceTier.MEDIUM if confidence >= 0.4 else ConfidenceTier.LOW
    return GRNSimulationResult(
        organism=organism, perturbation_gene=perturbation,
        knockdown_fraction=0.1, genes=[perturbation] + [f"gene{i}" for i in range(n_total)],
        edges=edges, n_cited_edges=n_cited, n_total_edges=n_total,
        confidence_score=confidence, confidence_tier=tier,
        timepoints=[0.0, 1.0], trajectories={perturbation: [0.1, 0.1]},
        unknown_parameters=unknown, warnings=[], notes=[],
    )


def _bio_result(channel_class="K", cited=True):
    return BioelectricSimulationResult(
        channel_class=channel_class, channel_class_cited=cited, max_dm_fraction=0.1,
        baseline_vmem_mV=-43.0, perturbed_vmem_mV=-9.4, delta_vmem_mV=33.6,
        cell_count=225, runtime_seconds=60.0, warnings=[], notes=[],
    )


# ======================================================================
# Mode detection
# ======================================================================
def test_detect_mode_prediction_triggers():
    assert he.detect_mode("Predict the effect of knocking down yugO") == NexusMode.PREDICTION
    assert he.detect_mode("What happens if we knock down sigB?") == NexusMode.PREDICTION
    assert he.detect_mode("Simulate the effect of a kinC knockout") == NexusMode.PREDICTION


def test_get_mode_prompt_prediction():
    prompt = he.get_mode_prompt(NexusMode.PREDICTION)
    assert "MODE 6" in prompt
    assert "Predicted From Cited Data" in prompt


# ======================================================================
# Confidence math: exactly (cited GRN edges + cited bio mapping) / (total
# GRN edges + 1 if bioelectric ran) -- never a separate judgment call.
# ======================================================================
def test_confidence_combines_grn_and_cited_bioelectric_leg(monkeypatch):
    monkeypatch.setattr(he, "simulate_grn", lambda *a, **k: _grn_result(n_cited=1, n_total=1))
    monkeypatch.setattr(he, "simulate_bioelectric", lambda *a, **k: _bio_result(cited=True))

    result = he.run_prediction_mode("B. subtilis", "yugO")

    # 1 cited GRN edge + 1 cited bioelectric mapping = 2 cited / 2 total = 1.0
    assert result.overall_confidence_score == pytest.approx(1.0)
    assert result.overall_confidence_tier == ConfidenceTier.HIGH


def test_confidence_penalizes_uncited_bioelectric_mapping(monkeypatch):
    monkeypatch.setattr(he, "simulate_grn", lambda *a, **k: _grn_result(n_cited=1, n_total=1))
    monkeypatch.setattr(he, "simulate_bioelectric", lambda *a, **k: _bio_result(cited=False))

    result = he.run_prediction_mode("B. subtilis", "yugO")

    # 1 cited GRN edge + 0 cited bioelectric mapping = 1 cited / 2 total = 0.5
    assert result.overall_confidence_score == pytest.approx(0.5)
    assert result.overall_confidence_tier == ConfidenceTier.MEDIUM


def test_confidence_grn_only_when_bioelectric_unavailable(monkeypatch):
    monkeypatch.setattr(he, "simulate_grn", lambda *a, **k: _grn_result(n_cited=1, n_total=2))

    def _raise(*a, **k):
        raise BETSEBackendUnavailable()
    monkeypatch.setattr(he, "simulate_bioelectric", _raise)

    result = he.run_prediction_mode("B. subtilis", "yugO")

    # No bioelectric parameter added to the denominator: 1/2 = 0.5
    assert result.overall_confidence_score == pytest.approx(0.5)
    assert result.bioelectric is None
    assert any("Bioelectric leg skipped" in w for w in result.warnings)


def test_confidence_zero_total_params_is_zero_not_error(monkeypatch):
    monkeypatch.setattr(he, "simulate_grn", lambda *a, **k: _grn_result(n_cited=0, n_total=0))

    def _raise(*a, **k):
        raise BETSEBackendUnavailable()
    monkeypatch.setattr(he, "simulate_bioelectric", _raise)

    result = he.run_prediction_mode("B. subtilis", "yugO")
    assert result.overall_confidence_score == 0.0
    assert result.overall_confidence_tier == ConfidenceTier.LOW


# ======================================================================
# Cited vs. assumed split
# ======================================================================
def test_cited_and_assumed_predictions_are_separated(monkeypatch):
    monkeypatch.setattr(he, "simulate_grn", lambda *a, **k: _grn_result(n_cited=1, n_total=2))
    monkeypatch.setattr(he, "simulate_bioelectric", lambda *a, **k: _bio_result(cited=True))

    result = he.run_prediction_mode("B. subtilis", "yugO")

    assert len(result.cited_predictions) == 2  # 1 GRN edge + 1 bioelectric mapping
    assert len(result.assumed_predictions) == 1  # 1 uncited GRN edge
    assert all("cited" in c or "BETSE" in c for c in result.cited_predictions)
    assert "no cited rate/affinity" in result.assumed_predictions[0]


def test_uncited_bioelectric_mapping_goes_to_assumed(monkeypatch):
    monkeypatch.setattr(he, "simulate_grn", lambda *a, **k: _grn_result(n_cited=1, n_total=1))
    monkeypatch.setattr(he, "simulate_bioelectric", lambda *a, **k: _bio_result(cited=False))

    result = he.run_prediction_mode("B. subtilis", "yugO")

    assert len(result.cited_predictions) == 1  # only the GRN edge
    assert any("not grounded in cited evidence" in a for a in result.assumed_predictions)


# ======================================================================
# experiment_designer.py feed + organism-mismatch caveat
# ======================================================================
def test_experiment_protocol_flags_organism_mismatch(monkeypatch):
    monkeypatch.setattr(he, "simulate_grn", lambda *a, **k: _grn_result(n_cited=1, n_total=1))
    monkeypatch.setattr(he, "simulate_bioelectric", lambda *a, **k: _bio_result(cited=True))

    result = he.run_prediction_mode("B. subtilis", "yugO")

    assert result.experiment_protocol is not None
    assert "MINIMAL WET-LAB TEST" in result.experiment_protocol
    assert result.experiment_protocol_caveat is not None
    assert "B. subtilis" in result.experiment_protocol_caveat


def test_no_protocol_proposed_when_bioelectric_did_not_run(monkeypatch):
    monkeypatch.setattr(he, "simulate_grn", lambda *a, **k: _grn_result(n_cited=1, n_total=1))

    def _raise(*a, **k):
        raise BETSEBackendUnavailable()
    monkeypatch.setattr(he, "simulate_bioelectric", _raise)

    result = he.run_prediction_mode("B. subtilis", "yugO")

    assert result.experiment_protocol is None
    assert "did not run" in result.experiment_protocol_caveat


def test_organism_matches_helper():
    assert he._organism_matches("Schmidtea mediterranea (asexual CIW4)", "S. mediterranea")
    assert he._organism_matches("B. subtilis", "B. subtilis")
    assert not he._organism_matches("Schmidtea mediterranea", "B. subtilis")


# ======================================================================
# format_prediction_context -- braces in content must not break the later
# pipeline.py .format(context=..., query=...) call on the query template.
# ======================================================================
def test_get_mode_query_template_escapes_braces_in_prediction_context():
    context = "some evidence mentions {a set} of genes"
    template = he.get_mode_query_template(NexusMode.PREDICTION, query="q", prediction_context=context)
    # Must not raise despite the literal braces in prediction_context, and
    # must not have substituted them as if they were format fields.
    rendered = template.format(context="ctx", query="q")
    assert "{a set}" in rendered
