"""Tests for the SubtiWiki collector.

Offline parsing tests use fixture dicts shaped exactly like SubtiWiki's real
API responses (field names taken from the live swagger.json spec at
https://subtiwiki.uni-goettingen.de/v5/api/swagger.json, captured 2026-09-04
— not guessed), matching the offline style of test_bio_collectors.py.

The ChromaDB test ingests a small sample of gene records end-to-end
(Paper -> TextChunk -> VectorStore) into a temporary on-disk collection and
confirms they land correctly, including the new organism/source_type fields.
"""

import tempfile
from pathlib import Path

from acheron.collectors.subtiwiki import SubtiWikiCollector
from acheron.extraction.chunker import TextChunker
from acheron.models import PaperSource, SourceType
from acheron.vectorstore.store import VectorStore

# ======================================================================
# Fixtures shaped like real SubtiWiki API responses (Gene schema)
# ======================================================================
SIGB_GENE_FIXTURE = {
    "id": 1801,
    "name": "sigB",
    "description": "general stress sigma factor",
    "function": "general stress response",
    "product": "RNA polymerase sigma factor SigB",
    "essential": "no",
    "mutant_phenotypes": ["reduced survival under stress conditions"],
    "expression_and_regulation": ["induced by heat, salt, and ethanol stress"],
    "genomic_annotations": [
        {"locus_tag": "BSU04730", "start": 502285, "end": 503094, "orientation": "Positive"}
    ],
    "protein": {
        "interactions": [
            {
                "molecules": [{"id": 1, "name": "SigB"}, {"id": 2, "name": "RsbW"}],
                "description": "RsbW binds and inhibits SigB",
            }
        ]
    },
    "regulations": [
        {
            "regulon": {"regulator_display_name": None, "regulator_gene": {"id": 55, "name": "RsbW"}},
            "mechanism": "SigmaFactor",
            "mode": "Negative",
            "description": "anti-sigma factor",
        }
    ],
    "regulon": {
        "regulator_display_name": "SigB",
        "gene_regulations": [{"gene": {"id": 10, "name": "ctc"}}, {"gene": {"id": 11, "name": "gsiB"}}],
        "operon_regulations": [],
    },
    "references": [{"pubmed_id": 8202364, "type": "Reviews"}],
}

# A minimal, sparse gene record (most optional fields absent) — exercises
# that the parser never invents missing data, it just omits those lines.
MINIMAL_GENE_FIXTURE = {
    "id": 42,
    "name": "yqzE",
    "genomic_annotations": [],
}

EXPRESSION_FIXTURE = [
    {"condition": {"id": 1, "name": "Heat stress (48C, 10 min)"}, "gene_id": 1801, "value": 12.4},
    {"condition": {"id": 2, "name": "Salt stress (1.2M NaCl)"}, "gene_id": 1801, "value": 8.9},
]


def test_parse_gene_full_record():
    collector = SubtiWikiCollector.__new__(SubtiWikiCollector)  # skip __init__ (no network needed)
    paper = collector._parse_gene(SIGB_GENE_FIXTURE, EXPRESSION_FIXTURE)

    assert paper is not None
    assert paper.paper_id == "subtiwiki:gene:1801"
    assert paper.source == PaperSource.SUBTIWIKI
    assert paper.organism == "B. subtilis"
    assert paper.source_type == SourceType.CURATED_DB
    assert "BSU04730" in paper.abstract
    assert "general stress response" in paper.abstract
    assert "RsbW" in paper.abstract  # protein-protein interaction partner
    assert "Regulated by RsbW" in paper.abstract  # regulon relationship
    assert "regulates 2 gene(s)" in paper.abstract  # this gene as a regulator
    assert "Heat stress" in paper.abstract  # expression-by-condition data
    assert "8202364" in paper.abstract  # SubtiWiki's own PubMed reference


def test_parse_gene_minimal_record_no_invented_fields():
    collector = SubtiWikiCollector.__new__(SubtiWikiCollector)
    paper = collector._parse_gene(MINIMAL_GENE_FIXTURE, [])

    assert paper is not None
    assert paper.paper_id == "subtiwiki:gene:42"
    # No locus tag was present in the source data — the parser must say so
    # plainly rather than inventing one.
    assert "not recorded in SubtiWiki" in paper.abstract
    assert "Interacts with" not in paper.abstract
    assert "Regulated by" not in paper.abstract
    assert "Expression by condition" not in paper.abstract


def test_parse_gene_missing_id_or_name_returns_none():
    collector = SubtiWikiCollector.__new__(SubtiWikiCollector)
    assert collector._parse_gene({}, []) is None
    assert collector._parse_gene({"id": 1}, []) is None  # no name
    assert collector._parse_gene({"name": "x"}, []) is None  # no id


def test_unwrap_envelope():
    collector = SubtiWikiCollector.__new__(SubtiWikiCollector)

    class FakeResponse:
        def json(self):
            return {"code": 200, "is_success": True, "data": {"id": 1801, "name": "sigB"}, "message": None}

    assert collector._unwrap(FakeResponse()) == {"id": 1801, "name": "sigB"}


def test_unwrap_envelope_failure_returns_none():
    collector = SubtiWikiCollector.__new__(SubtiWikiCollector)

    class FakeFailedResponse:
        def json(self):
            return {"code": 404, "is_success": False, "data": None, "message": "not found"}

    assert collector._unwrap(FakeFailedResponse()) is None


# ======================================================================
# ChromaDB ingestion — 5-10 sample genes, end to end
# ======================================================================
SAMPLE_GENES = [
    {"id": 1, "name": "dnaA", "function": "DNA replication initiation",
     "genomic_annotations": [{"locus_tag": "BSU00010"}]},
    {"id": 2, "name": "sigB", "function": "general stress response",
     "genomic_annotations": [{"locus_tag": "BSU04730"}]},
    {"id": 3, "name": "comK", "function": "competence transcription factor",
     "genomic_annotations": [{"locus_tag": "BSU24730"}]},
    {"id": 4, "name": "spo0A", "function": "master regulator of sporulation",
     "genomic_annotations": [{"locus_tag": "BSU24220"}]},
    {"id": 5, "name": "sinR", "function": "biofilm repressor",
     "genomic_annotations": [{"locus_tag": "BSU24260"}]},
    {"id": 6, "name": "degU", "function": "response regulator",
     "genomic_annotations": [{"locus_tag": "BSU21830"}]},
    {"id": 7, "name": "abrB", "function": "transition state regulator",
     "genomic_annotations": [{"locus_tag": "BSU39130"}]},
]


def test_sample_genes_ingest_into_chromadb():
    collector = SubtiWikiCollector.__new__(SubtiWikiCollector)
    papers = [collector._parse_gene(g, []) for g in SAMPLE_GENES]
    assert len(papers) == len(SAMPLE_GENES) == 7
    assert all(p is not None for p in papers)

    chunker = TextChunker()
    all_chunks = []
    for paper in papers:
        all_chunks.extend(chunker.chunk_paper(paper))
    assert len(all_chunks) >= len(papers)  # at least one chunk per gene

    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(persist_dir=Path(tmp))
        added = store.add_chunks(all_chunks)
        assert added == len(all_chunks)
        assert store.count() == len(all_chunks)

        # Re-adding the same chunks must not duplicate them (paper_id-based dedup).
        added_again = store.add_chunks(all_chunks)
        assert added_again == 0
        assert store.count() == len(all_chunks)

        results = store.search("sporulation", n_results=5)
        assert len(results) > 0
        assert any("spo0A" in r.text or "spo0A" in (r.paper_title or "") for r in results)
        matched = next(r for r in results if "spo0A" in r.text or "spo0A" in (r.paper_title or ""))
        assert matched.source == PaperSource.SUBTIWIKI.value or matched.source == PaperSource.SUBTIWIKI
