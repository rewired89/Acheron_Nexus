"""Tests for predicted-vs-actual tracking in rag/ledger.py: record_prediction(),
record_actual_outcome(), prediction_entries(), and compute_accuracy_report().

Offline and fast -- ExperimentLedger just writes/reads JSON files under a
temp directory, no network or LLM calls, matching this project's existing
test_ledger.py pattern.
"""

import pytest

from acheron.rag.ledger import ExperimentLedger, compute_accuracy_report


def _log_prediction(store, organism="B. subtilis", gene="yugO", cited=1, total=2, tier="medium"):
    return store.record_prediction(
        organism=organism,
        perturbation_gene=gene,
        confidence_score=cited / total,
        confidence_tier=tier,
        cited_params=cited,
        total_params=total,
        prediction_summary="Organism: %s\nPerturbation: %s\n(fixture summary)" % (organism, gene),
    )


# ======================================================================
# record_prediction
# ======================================================================
def test_record_prediction_creates_prediction_type_entry(tmp_path):
    store = ExperimentLedger(tmp_path)
    entry = _log_prediction(store)

    assert entry.entry_type == "prediction"
    assert entry.organism == "B. subtilis"
    assert entry.perturbation_gene == "yugO"
    assert entry.predicted_confidence_score == pytest.approx(0.5)
    assert entry.predicted_confidence_tier == "medium"
    assert entry.predicted_cited_params == 1
    assert entry.predicted_total_params == 2
    assert entry.actual_outcome is None
    assert entry.match is None

    reloaded = store.get_entry(entry.entry_id)
    assert reloaded.entry_type == "prediction"
    assert reloaded.organism == "B. subtilis"


def test_record_prediction_does_not_appear_as_resolved(tmp_path):
    store = ExperimentLedger(tmp_path)
    _log_prediction(store)
    assert store.prediction_entries(resolved_only=False) != []
    assert store.prediction_entries(resolved_only=True) == []


# ======================================================================
# record_actual_outcome
# ======================================================================
def test_record_actual_outcome_attaches_result(tmp_path):
    store = ExperimentLedger(tmp_path)
    entry = _log_prediction(store)

    updated = store.record_actual_outcome(entry.entry_id, "Real Vmem shifted +30mV", match=True)

    assert updated.actual_outcome == "Real Vmem shifted +30mV"
    assert updated.match is True
    assert updated.actual_outcome_entered_at is not None
    # The original prediction timestamp must be preserved, proving the
    # prediction was logged before the outcome.
    assert updated.timestamp < updated.actual_outcome_entered_at


def test_record_actual_outcome_refuses_unknown_entry(tmp_path):
    store = ExperimentLedger(tmp_path)
    with pytest.raises(ValueError, match="No ledger entry found"):
        store.record_actual_outcome("ledger-does-not-exist", "x", match=True)


def test_record_actual_outcome_refuses_non_prediction_entry(tmp_path):
    from acheron.models import DiscoveryResult

    store = ExperimentLedger(tmp_path)
    entry = store.record(DiscoveryResult(query="a plain discovery query"))

    with pytest.raises(ValueError, match="not a prediction"):
        store.record_actual_outcome(entry.entry_id, "x", match=True)


def test_record_actual_outcome_refuses_silent_overwrite(tmp_path):
    store = ExperimentLedger(tmp_path)
    entry = _log_prediction(store)
    store.record_actual_outcome(entry.entry_id, "first result", match=True)

    with pytest.raises(ValueError, match="already has an actual outcome"):
        store.record_actual_outcome(entry.entry_id, "second result", match=False)

    # force=True permits the overwrite.
    forced = store.record_actual_outcome(entry.entry_id, "second result", match=False, force=True)
    assert forced.actual_outcome == "second result"
    assert forced.match is False


def test_prediction_entries_resolved_only_after_outcome_recorded(tmp_path):
    store = ExperimentLedger(tmp_path)
    entry = _log_prediction(store)
    assert store.prediction_entries(resolved_only=True) == []

    store.record_actual_outcome(entry.entry_id, "result", match=True)
    resolved = store.prediction_entries(resolved_only=True)
    assert len(resolved) == 1
    assert resolved[0].entry_id == entry.entry_id


# ======================================================================
# compute_accuracy_report -- the investor-facing metric's actual math
# ======================================================================
def test_accuracy_report_empty_when_nothing_resolved(tmp_path):
    store = ExperimentLedger(tmp_path)
    _log_prediction(store)  # logged but not resolved

    report = compute_accuracy_report(store.prediction_entries())
    assert report["resolved_count"] == 0
    assert report["overall"] is None
    assert "No real forward-tested predictions" in report["note"]


def test_accuracy_report_ignores_discovery_entries(tmp_path):
    from acheron.models import DiscoveryResult

    store = ExperimentLedger(tmp_path)
    store.record(DiscoveryResult(query="unrelated discovery-mode query"))

    report = compute_accuracy_report(store.prediction_entries())
    assert report["resolved_count"] == 0


def test_accuracy_report_computes_overall_and_breakdowns(tmp_path):
    store = ExperimentLedger(tmp_path)

    e1 = _log_prediction(store, organism="B. subtilis", gene="yugO", cited=2, total=2, tier="high")
    store.record_actual_outcome(e1.entry_id, "matched", match=True)

    e2 = _log_prediction(store, organism="B. subtilis", gene="kinC", cited=0, total=2, tier="low")
    store.record_actual_outcome(e2.entry_id, "did not match", match=False)

    e3 = _log_prediction(store, organism="S. mediterranea", gene="inx-6", cited=1, total=2, tier="medium")
    store.record_actual_outcome(e3.entry_id, "matched", match=True)

    e4 = _log_prediction(store, organism="B. subtilis", gene="sigB", cited=1, total=1, tier="high")
    # e4 left unresolved -- must not affect the accuracy figure at all.

    report = compute_accuracy_report(store.prediction_entries())

    assert report["resolved_count"] == 3
    assert report["overall"]["n"] == 3
    assert report["overall"]["matches"] == 2
    assert report["overall"]["accuracy_pct"] == pytest.approx(66.7)

    assert report["by_organism"]["B. subtilis"]["n"] == 2
    assert report["by_organism"]["B. subtilis"]["matches"] == 1
    assert report["by_organism"]["S. mediterranea"]["n"] == 1
    assert report["by_organism"]["S. mediterranea"]["matches"] == 1

    assert report["by_confidence_tier"]["high"]["n"] == 1
    assert report["by_confidence_tier"]["high"]["accuracy_pct"] == 100.0
    assert report["by_confidence_tier"]["low"]["n"] == 1
    assert report["by_confidence_tier"]["low"]["accuracy_pct"] == 0.0
    assert report["by_confidence_tier"]["medium"]["n"] == 1
