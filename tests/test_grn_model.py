"""Tests for the GRN simulation backend (src/acheron/simulation/grn_model.py).

Offline, fixture-based, following the pattern in test_parameter_extractor.py:
ParameterRecords are built directly (this module consumes ParameterRecords,
not raw chunks, so there's no need to go through the chunker/extractor here).

The end-to-end simulation tests actually run libRoadRunner/Antimony and are
skipped (not failed) if those packages aren't installed, since they are an
optional extra (`pip install -e ".[simulation]"`), not a hard dependency of
the rest of Acheron Nexus.
"""

import pytest

from acheron.extraction.parameter_extractor import ConfidenceTier, ParameterRecord, ParameterStore
from acheron.simulation import grn_model as grn


def _record(
    subject: str,
    relationship_type: str,
    obj: str,
    rate_or_affinity: str = "UNKNOWN",
    evidence_text: str = "",
    organism: str = "B. subtilis",
    confidence_score: float = 0.5,
    confidence_tier: ConfidenceTier = ConfidenceTier.MEDIUM,
    record_id: str = None,
) -> ParameterRecord:
    return ParameterRecord(
        record_id=record_id or f"param-{subject}-{relationship_type}-{obj}",
        subject=subject,
        relationship_type=relationship_type,
        object=obj,
        rate_or_affinity=rate_or_affinity,
        evidence_text=evidence_text or f"{subject} {relationship_type} {obj}",
        organism=organism,
        source_type="curated_db",
        source="subtiwiki",
        paper_id=f"subtiwiki:gene:{subject}",
        chunk_id=f"chunk-{subject}",
        confidence_tier=confidence_tier,
        confidence_score=confidence_score,
    )


# ======================================================================
# Sign inference
# ======================================================================
def test_infer_sign_explicit_from_relationship_type():
    assert grn._infer_sign("activates", "") == ("activation", True)
    assert grn._infer_sign("inhibits", "") == ("repression", True)


def test_infer_sign_from_evidence_text_keyword():
    assert grn._infer_sign("regulated_by", "Regulated by RsbW (sigma factor, negative)") == (
        "repression", True,
    )
    assert grn._infer_sign("regulated_by", "Regulated by RsbU (positive): activates sigB") == (
        "activation", True,
    )


def test_infer_sign_defaults_and_flags_when_unstated():
    sign, cited = grn._infer_sign("interacts_with", "Interacts with RsbV")
    assert sign == "activation"
    assert cited is False


# ======================================================================
# Rate constant resolution -- no invented numbers
# ======================================================================
def test_resolve_rate_constant_unknown_uses_placeholder_and_flags_uncited():
    rate, cited = grn._resolve_rate_constant("UNKNOWN")
    assert rate == grn.DEFAULT_RATE_CONSTANT
    assert cited is False


def test_resolve_rate_constant_uses_literal_cited_number():
    rate, cited = grn._resolve_rate_constant("Kd = 12 nM")
    assert rate == 12.0
    assert cited is True


# ======================================================================
# Record -> edge conversion and filtering
# ======================================================================
def test_edge_from_record_regulated_by_reverses_direction():
    record = _record("sigB", "regulated_by", "RsbW", evidence_text="Regulated by RsbW (negative)")
    edge = grn._edge_from_record(record)
    assert edge["source_gene"] == "RsbW"
    assert edge["target_gene"] == "sigB"
    assert edge["directed"] is True
    assert edge["sign"] == "repression"
    assert edge["relationship_class"] == "regulatory"


def test_edge_from_record_interacts_with_is_undirected_physical():
    record = _record("sigB", "interacts_with", "RsbV")
    edge = grn._edge_from_record(record)
    assert edge["directed"] is False
    assert edge["relationship_class"] == "physical_interaction"


def test_edge_from_record_excludes_regulon_summary():
    record = _record(
        "sigB", "regulates",
        "150 gene(s), 12 operon(s) (regulon summary, not individual targets)",
    )
    assert grn._edge_from_record(record) is None


def test_edge_from_record_excludes_non_grn_relationship_types():
    record = _record("smedwi-1", "homologous_to", "FBgn0027107")
    assert grn._edge_from_record(record) is None


# ======================================================================
# Dedup
# ======================================================================
def test_dedup_prefers_cited_rate_over_uncited():
    uncited = grn._edge_from_record(_record("sigB", "activates", "ctc", rate_or_affinity="UNKNOWN"))
    cited = grn._edge_from_record(
        _record("sigB", "activates", "ctc", rate_or_affinity="Kd = 5 nM", record_id="param-cited")
    )
    deduped = grn._dedup_edges([uncited, cited])
    assert len(deduped) == 1
    assert deduped[0]["rate_cited"] is True
    assert deduped[0]["rate_constant"] == 5.0


# ======================================================================
# Downstream network BFS
# ======================================================================
def test_build_downstream_network_respects_max_hops():
    edges = [
        grn._edge_from_record(_record("A", "activates", "B")),
        grn._edge_from_record(_record("B", "activates", "C")),
        grn._edge_from_record(_record("C", "activates", "D")),
    ]
    visited, included = grn._build_downstream_network(edges, "a", max_hops=1)
    assert visited == {"a", "b"}
    assert len(included) == 1

    visited2, included2 = grn._build_downstream_network(edges, "a", max_hops=2)
    assert visited2 == {"a", "b", "c"}
    assert len(included2) == 2


def test_build_downstream_network_isolated_gene_has_no_edges():
    edges = [grn._edge_from_record(_record("A", "activates", "B"))]
    visited, included = grn._build_downstream_network(edges, "z", max_hops=2)
    assert visited == {"z"}
    assert included == []


# ======================================================================
# Organism alias resolution
# ======================================================================
def test_resolve_organism_aliases():
    assert grn._resolve_organism("bacteria") == "B. subtilis"
    assert grn._resolve_organism("Bacillus subtilis") == "B. subtilis"
    assert grn._resolve_organism("planarian") == "S. mediterranea"
    assert grn._resolve_organism("Some Other Organism") == "Some Other Organism"


# ======================================================================
# End-to-end simulation (skipped if antimony/libroadrunner aren't installed)
# ======================================================================
def test_simulate_grn_no_records_raises_value_error(tmp_path):
    store = ParameterStore(tmp_path)
    store.count()  # ensures dir exists, still empty
    with pytest.raises(ValueError, match="No parameter records found"):
        grn.simulate_grn("B. subtilis", "yugO", params_dir=tmp_path)


def test_simulate_grn_end_to_end(tmp_path):
    pytest.importorskip("roadrunner")
    pytest.importorskip("antimony")

    store = ParameterStore(tmp_path)
    store.save_all([
        _record("yugO", "regulated_by", "SinR", evidence_text="Regulated by SinR (negative)"),
        _record(
            "yugO", "activates", "kinC", rate_or_affinity="UNKNOWN",
            evidence_text="YugO K+ efflux activates KinC",
        ),
    ])

    result = grn.simulate_grn(
        "bacteria", "yugO", knockdown_fraction=0.1, duration=50.0, n_points=6,
        params_dir=tmp_path,
    )

    assert result.organism == "B. subtilis"
    assert "yugO" in result.genes
    assert "kinC" in result.genes
    assert result.n_total_edges == 2
    # Neither edge in this fixture carries a cited rate constant.
    assert result.n_cited_edges == 0
    assert result.confidence_score == 0.0
    assert result.confidence_tier == ConfidenceTier.LOW
    assert len(result.unknown_parameters) >= 2  # both rates, at least one sign

    # Perturbed gene held fixed at knockdown_fraction * baseline throughout.
    assert result.trajectories["yugO"] == pytest.approx([0.1] * len(result.timepoints))
    # Downstream gene should move away from baseline (1.0) since it's driven
    # by the perturbed gene's deviation from baseline.
    assert result.trajectories["kinC"][-1] != pytest.approx(1.0)


def test_simulate_grn_perturbation_gene_not_in_store_warns(tmp_path):
    pytest.importorskip("roadrunner")
    pytest.importorskip("antimony")

    store = ParameterStore(tmp_path)
    store.save_all([_record("sigB", "activates", "ctc")])

    result = grn.simulate_grn("B. subtilis", "unknownGeneXYZ", params_dir=tmp_path)
    assert result.n_total_edges == 0
    assert result.confidence_score == 0.0
    assert any("does not appear" in w for w in result.warnings)
    assert result.trajectories["unknownGeneXYZ"] == pytest.approx(
        [grn.BASELINE_EXPRESSION * grn.DEFAULT_KNOCKDOWN_FRACTION] * len(result.timepoints)
    )
