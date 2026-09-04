"""Tests for the PlanMine collector.

Offline parsing tests use fixture data shaped exactly like PlanMine's real
schema (class/attribute names taken from PlanMine's live GET .../service/model
response, captured 2026-09-04 — not guessed), matching the offline style of
test_bio_collectors.py / test_subtiwiki_collector.py. InterMine's REST API
returns query results as rows (lists) in view-column order, not keyed
dicts — these fixtures use that same row shape.
"""

import tempfile
from pathlib import Path

from acheron.collectors.planmine import PlanMineCollector, _pathquery
from acheron.extraction.chunker import TextChunker
from acheron.models import PaperSource, SourceType
from acheron.vectorstore.store import VectorStore

# ======================================================================
# Fixtures shaped like real PlanMine/InterMine row-based query results
# ======================================================================
GENE_VIEW = [
    "Gene.primaryIdentifier",
    "Gene.secondaryIdentifier",
    "Gene.symbol",
    "Gene.briefDescription",
    "Gene.description",
    "Gene.organism.name",
    "Gene.organism.shortName",
    "Gene.organism.commonName",
]

INNEXIN_GENE_ROW = dict(zip(GENE_VIEW, [
    "SMED30033840", None, "Smed-inx-11", "innexin family gap junction protein",
    "gap junction channel protein expressed in neoblasts",
    "Schmidtea mediterranea", "S. mediterranea", "planarian",
]))

MINIMAL_GENE_ROW = dict(zip(GENE_VIEW, [
    "SMED30011111", None, None, None, None, None, None, None,
]))

HOMOLOGY_ROWS = [["orthologue", "FBgn0027107"], ["paralogue", "SMED30099999"]]

RNAI_ROWS = [
    [3.2, "reduced regeneration", "amputation, 7 days", "Reddien lab RNAi screen", 21778508],
]

DOMAIN_ROWS = [
    ["Innexin", "PF00876"],
    ["ATP-binding domain", "PF00005"],
]


def test_parse_gene_full_record_flags_ion_channel():
    collector = PlanMineCollector.__new__(PlanMineCollector)  # skip __init__, no network needed
    paper = collector._parse_gene(INNEXIN_GENE_ROW, HOMOLOGY_ROWS, RNAI_ROWS, DOMAIN_ROWS)

    assert paper is not None
    assert paper.paper_id == "planmine:SMED30033840"
    assert paper.source == PaperSource.PLANMINE
    assert paper.organism == "S. mediterranea"
    assert paper.source_type == SourceType.CURATED_DB
    assert "gap junction channel protein" in paper.abstract
    assert "not curated" in paper.abstract  # symbol caveat always present when a symbol exists
    assert "FBgn0027107" in paper.abstract  # homology annotation
    assert "reduced regeneration" in paper.abstract  # RNAi result
    assert "NOT a curated phenotype" in paper.abstract  # sparse-data flag always present with RNAi data
    assert "ION CHANNEL / GAP JUNCTION RELEVANT" in paper.abstract
    assert "Innexin" in paper.abstract


def test_parse_gene_minimal_record_no_invented_fields():
    collector = PlanMineCollector.__new__(PlanMineCollector)
    paper = collector._parse_gene(MINIMAL_GENE_ROW, [], [], [])

    assert paper is not None
    assert paper.paper_id == "planmine:SMED30011111"
    # No symbol, no homology, no RNAi, no domains — none of those section
    # headers should appear when the source genuinely returned nothing.
    assert "Symbol:" not in paper.abstract
    assert "Homology annotations" not in paper.abstract
    assert "RNAi/RNA-seq" not in paper.abstract
    assert "Protein domain annotations" not in paper.abstract
    assert "ION CHANNEL" not in paper.abstract


def test_parse_gene_missing_id_returns_none():
    collector = PlanMineCollector.__new__(PlanMineCollector)
    assert collector._parse_gene({}, [], [], []) is None
    assert collector._parse_gene({"Gene.primaryIdentifier": None}, [], [], []) is None


def test_pathquery_builds_valid_xml():
    xml = _pathquery(["Gene.primaryIdentifier", "Gene.symbol"], "Gene.symbol", "CONTAINS", "inx-11")
    assert 'path="Gene.symbol"' in xml
    assert 'op="CONTAINS"' in xml
    assert 'value="inx-11"' in xml
    assert 'view="Gene.primaryIdentifier Gene.symbol"' in xml

    # Must be well-formed XML — parse it back to confirm.
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    assert root.tag == "query"


def test_run_query_missing_results_key_returns_empty():
    collector = PlanMineCollector.__new__(PlanMineCollector)

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"unexpected": "shape"}

    class FakeClient:
        def get(self, url, **kwargs):
            return FakeResponse()

    collector.client = FakeClient()
    rows = collector._run_query(["Gene.primaryIdentifier"], "Gene.primaryIdentifier", "=", "X")
    assert rows == []


# ======================================================================
# ChromaDB ingestion — small sample of genes, same pattern as Phase 1
# ======================================================================
SAMPLE_GENES = [
    (dict(zip(GENE_VIEW, ["SMED1", None, "smedwi-1", "neoblast marker", "stem cell marker gene",
                           "Schmidtea mediterranea", "S. mediterranea", "planarian"])), [], [], []),
    (dict(zip(GENE_VIEW, ["SMED2", None, "djnos", "nitric oxide synthase", None,
                           "Schmidtea mediterranea", "S. mediterranea", "planarian"])), [], [], []),
    (dict(zip(GENE_VIEW, ["SMED3", None, "Smed-inx-11", "innexin gap junction protein", None,
                           "Schmidtea mediterranea", "S. mediterranea", "planarian"])),
     [], [], [["Innexin", "PF00876"]]),
    (dict(zip(GENE_VIEW, ["SMED4", None, "Smed-inx-13", "innexin gap junction protein", None,
                           "Schmidtea mediterranea", "S. mediterranea", "planarian"])),
     [], [], [["Innexin", "PF00876"]]),
    (dict(zip(GENE_VIEW, ["SMED5", None, None, None, "uncharacterized ORF",
                           "Schmidtea mediterranea", "S. mediterranea", "planarian"])), [], [], []),
    (dict(zip(GENE_VIEW, ["SMED6", None, "djpc2", "piwi-family gene", None,
                           "Schmidtea mediterranea", "S. mediterranea", "planarian"])),
     [], [[1.1, "no visible phenotype", "control", "Reddien lab RNAi screen", 21778508]], []),
]


def test_sample_genes_ingest_into_chromadb():
    collector = PlanMineCollector.__new__(PlanMineCollector)
    papers = [collector._parse_gene(*sample) for sample in SAMPLE_GENES]
    assert len(papers) == len(SAMPLE_GENES) == 6
    assert all(p is not None for p in papers)

    chunker = TextChunker()
    all_chunks = []
    for paper in papers:
        all_chunks.extend(chunker.chunk_paper(paper))
    assert len(all_chunks) >= len(papers)

    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(persist_dir=Path(tmp))
        added = store.add_chunks(all_chunks)
        assert added == len(all_chunks)
        assert store.count() == len(all_chunks)

        added_again = store.add_chunks(all_chunks)
        assert added_again == 0

        results = store.search("gap junction innexin protein", n_results=5)
        assert len(results) > 0
        assert any("inx-1" in r.text for r in results)
