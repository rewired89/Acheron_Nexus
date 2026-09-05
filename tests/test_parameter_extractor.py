"""Tests for the parameter extractor (Phase 3).

Offline, fixture-based, following the pattern in test_subtiwiki_collector.py
and test_planmine_collector.py: TextChunks are built directly (no live
ChromaDB/network calls needed except in the one small round-trip test at
the bottom, which mirrors the ChromaDB tests those two files already run).
"""

import tempfile
from pathlib import Path

from acheron.extraction.chunker import TextChunker
from acheron.extraction.parameter_extractor import (
    UNKNOWN,
    ConfidenceTier,
    ParameterStore,
    extract_from_chunk,
    extract_from_store,
)
from acheron.models import Paper, PaperSource, SourceType
from acheron.vectorstore.store import VectorStore


def _chunk_for(paper: Paper):
    chunks = TextChunker().chunk_paper(paper)
    assert len(chunks) == 1  # a short structured abstract stays a single chunk
    return chunks[0]


def _subtiwiki_paper() -> Paper:
    abstract = "\n".join([
        "Gene: sigB",
        "Organism: B. subtilis",
        "Locus tag: BSU_04730",
        "Function: general stress response sigma factor",
        "Protein-protein interactions:",
        "  - Interacts with RsbW, RsbV (anti-sigma factor partnership)",
        "Regulation:",
        "  - Regulated by RsbU (transcription factor, positive): activates sigB under stress",
        "Acts as a regulator: regulates 150 gene(s) and 12 operon(s) directly (per SubtiWiki's regulon record).",
    ])
    return Paper(
        paper_id="subtiwiki:gene:123",
        title="sigB (BSU_04730), SubtiWiki gene record",
        authors=["SubtiWiki (University of Göttingen)"],
        abstract=abstract,
        source=PaperSource.SUBTIWIKI,
        organism="B. subtilis",
        source_type=SourceType.CURATED_DB,
    )


def _planmine_paper() -> Paper:
    abstract = "\n".join([
        "Gene: SMED30033840",
        "Symbol: Smed-inx-11 (PlanMine-inferred by orthology, not curated)",
        "Organism: Schmidtea mediterranea",
        "Homology annotations: FBgn0027107 (orthologue); SMED30099999 (paralogue)",
        "RNAi/RNA-seq differential-expression results (NOT a curated phenotype, see module docstring):",
        "  - Reddien lab RNAi screen, score=3.2, amputation, 7 days, reduced regeneration (PubMed 21778508)",
        "Protein domain annotations: Innexin; ATP-binding domain",
        "ION CHANNEL / GAP JUNCTION RELEVANT (matched domain annotation): Innexin",
    ])
    return Paper(
        paper_id="planmine:SMED30033840",
        title="SMED30033840 (Smed-inx-11), PlanMine gene record",
        authors=["PlanMine (Max Planck Institute)"],
        abstract=abstract,
        source=PaperSource.PLANMINE,
        organism="S. mediterranea",
        source_type=SourceType.CURATED_DB,
    )


def _literature_paper() -> Paper:
    return Paper(
        paper_id="pubmed:99999",
        title="Innexin-6 regulates planarian regeneration",
        authors=["Someone et al."],
        abstract=(
            "In S. mediterranea, innexin-6 interacts with innexin-4 to form "
            "functional gap junctions. Kd = 12 nM was measured for the "
            "complex under standard buffer conditions."
        ),
        source=PaperSource.PUBMED,
        organism="S. mediterranea",
        source_type=SourceType.LITERATURE,
    )


# ======================================================================
# SubtiWiki extraction
# ======================================================================
def test_extract_subtiwiki_interactions_and_regulation():
    chunk = _chunk_for(_subtiwiki_paper())
    records = extract_from_chunk(chunk)

    types = {r.relationship_type for r in records}
    assert "interacts_with" in types
    assert "regulated_by" in types
    assert "regulates" in types

    interacts = [r for r in records if r.relationship_type == "interacts_with"]
    partners = {r.object for r in interacts}
    assert partners == {"RsbW", "RsbV"}
    assert all(r.subject == "sigB" for r in records)
    assert all(r.rate_or_affinity == UNKNOWN for r in records)  # SubtiWiki never cites kinetic constants
    assert all(r.source == "subtiwiki" for r in records)
    assert all(not r.heuristic for r in records)

    regulated = [r for r in records if r.relationship_type == "regulated_by"][0]
    assert regulated.object == "RsbU"

    regulon = [r for r in records if r.relationship_type == "regulates"][0]
    assert "150 gene(s)" in regulon.object


def test_subtiwiki_organism_match_scores_one_against_bacteria_tier():
    """science_filter._ORGANISM_TIERS now has a "bacteria" tier, and
    _target_organism_for() maps "B. subtilis" chunks to it, so SubtiWiki
    records score organism_match=1.0 against their own organism instead of
    being penalized for not matching a planarian-only scorer."""
    chunk = _chunk_for(_subtiwiki_paper())
    records = extract_from_chunk(chunk)
    assert all(r.confidence_breakdown["organism_match"] == 1.0 for r in records)


# ======================================================================
# PlanMine extraction
# ======================================================================
def test_extract_planmine_homology_rnai_and_ion_channel():
    chunk = _chunk_for(_planmine_paper())
    records = extract_from_chunk(chunk)

    types = {r.relationship_type for r in records}
    assert "homologous_to" in types
    assert "differential_expression_under" in types
    assert "has_domain_annotation" in types
    assert "has_domain" in types

    homology = [r for r in records if r.relationship_type == "homologous_to"]
    homology_objects = {r.object for r in homology}
    assert "FBgn0027107" in homology_objects
    assert "SMED30099999" in homology_objects

    ion_channel = [r for r in records if r.relationship_type == "has_domain_annotation"][0]
    assert ion_channel.object == "Innexin"

    domain = [r for r in records if r.relationship_type == "has_domain"][0]
    assert domain.object == "ATP-binding domain"  # Innexin excluded, already covered by ion-channel record

    rnai = [r for r in records if r.relationship_type == "differential_expression_under"][0]
    assert "score=3.2" in rnai.evidence_text
    assert rnai.rate_or_affinity == UNKNOWN  # an RNAi score is not a kinetic rate/affinity constant

    assert all(r.subject == "SMED30033840" for r in records)
    assert all(not r.heuristic for r in records)


def test_planmine_organism_matches_planarian_tier():
    """S. mediterranea text does match science_filter's "planarian" tier
    (unlike B. subtilis), so PlanMine records score organism_match=1.0."""
    chunk = _chunk_for(_planmine_paper())
    records = extract_from_chunk(chunk)
    assert all(r.confidence_breakdown["organism_match"] == 1.0 for r in records)


# ======================================================================
# Literature fallback
# ======================================================================
def test_extract_literature_heuristic_fallback():
    chunk = _chunk_for(_literature_paper())
    records = extract_from_chunk(chunk)

    assert len(records) >= 1
    assert all(r.heuristic for r in records)
    interacts = [r for r in records if r.relationship_type == "interacts_with"]
    assert interacts
    assert interacts[0].object == "innexin-4"
    # A real cited Kd should be picked up rather than left UNKNOWN.
    assert any(r.rate_or_affinity != UNKNOWN for r in records)


def test_no_invented_rate_when_none_cited():
    paper = Paper(
        paper_id="pubmed:11111",
        title="X interacts with Y, no kinetics reported",
        authors=["Nobody"],
        abstract="Protein X interacts with protein Y in a pulldown assay.",
        source=PaperSource.PUBMED,
        organism="planarian",
        source_type=SourceType.LITERATURE,
    )
    chunk = _chunk_for(paper)
    records = extract_from_chunk(chunk)
    assert records
    assert all(r.rate_or_affinity == UNKNOWN for r in records)


# ======================================================================
# Confidence tiering
# ======================================================================
def test_confidence_tier_buckets_match_score():
    chunk = _chunk_for(_planmine_paper())
    records = extract_from_chunk(chunk)
    for r in records:
        if r.confidence_score >= 0.7:
            assert r.confidence_tier == ConfidenceTier.HIGH
        elif r.confidence_score >= 0.4:
            assert r.confidence_tier == ConfidenceTier.MEDIUM
        else:
            assert r.confidence_tier == ConfidenceTier.LOW


# ======================================================================
# Storage — separate from ChromaDB, mirrors ledger.py's pattern.
# ======================================================================
def test_parameter_store_round_trip():
    chunk = _chunk_for(_subtiwiki_paper())
    records = extract_from_chunk(chunk)

    with tempfile.TemporaryDirectory() as tmp:
        store = ParameterStore(params_dir=Path(tmp))
        assert store.count() == 0
        store.save_all(records)
        assert store.count() == len(records)

        loaded = store.list_records(organism="B. subtilis")
        assert len(loaded) == len(records)
        assert store.list_records(organism="S. mediterranea") == []


# ======================================================================
# End-to-end: index into ChromaDB, then extract straight from the store
# (the actual path `scripts/validate_parameters.py` uses).
# ======================================================================
def test_extract_from_store_end_to_end():
    chunker = TextChunker()
    papers = [_subtiwiki_paper(), _planmine_paper(), _literature_paper()]
    chunks = []
    for paper in papers:
        chunks.extend(chunker.chunk_paper(paper))

    with tempfile.TemporaryDirectory() as vec_tmp, tempfile.TemporaryDirectory() as param_tmp:
        vec_store = VectorStore(persist_dir=Path(vec_tmp))
        vec_store.add_chunks(chunks)

        records = extract_from_store(vec_store, params_dir=Path(param_tmp))
        assert len(records) > 0

        param_store = ParameterStore(params_dir=Path(param_tmp))
        assert param_store.count() == len(records)

        organisms = {r.organism for r in records}
        assert "B. subtilis" in organisms
        assert "S. mediterranea" in organisms
