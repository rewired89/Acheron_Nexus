"""Offline tests for the UniProt / PDB / AlphaFold DB / NCBI Gene collectors.

Exercises only the pure parsing logic against fixture dicts shaped like the
real API responses — no network calls, matching the offline style of
test_nexus_ingest.py. If the live schema drifts, `acheron collect --source
<name>` will surface it; these tests guard against parsing bugs in this repo.
"""

from acheron.collectors.ncbi_gene import NCBIGeneCollector
from acheron.collectors.structures import AlphaFoldCollector, PDBCollector
from acheron.collectors.uniprot import UniProtCollector
from acheron.models import PaperSource


# ======================================================================
# UniProt
# ======================================================================
UNIPROT_ENTRY_FIXTURE = {
    "primaryAccession": "P17302",
    "uniProtkbId": "CXA1_RAT",
    "organism": {"scientificName": "Rattus norvegicus", "taxonId": 10116},
    "proteinDescription": {
        "recommendedName": {"fullName": {"value": "Gap junction alpha-1 protein"}}
    },
    "genes": [{"geneName": {"value": "Gja1"}}],
    "sequence": {"length": 382, "value": "MGDWSALGKLLDKVQAYSTAGGKVWLSVLFIFRILLLGTAVESAWGDEQ"},
    "comments": [
        {
            "commentType": "FUNCTION",
            "texts": [{"value": "Gap junction protein forming intercellular channels."}],
        },
        {
            "commentType": "SUBCELLULAR LOCATION",
            "subcellularLocations": [{"location": {"value": "Cell membrane"}}],
        },
    ],
    "keywords": [{"id": "KW-0303", "name": "Gap junction"}],
    "uniProtKBCrossReferences": [
        {"database": "PDB", "id": "2ZW3"},
        {"database": "GO", "id": "GO:0005921"},
    ],
    "entryAudit": {"lastAnnotationUpdateDate": "2023-05-03"},
}


def test_uniprot_parse_entry_basic_fields():
    collector = UniProtCollector.__new__(UniProtCollector)  # skip __init__ (no network/settings needed for parsing)
    paper = collector._parse_entry(UNIPROT_ENTRY_FIXTURE)

    assert paper is not None
    assert paper.paper_id == "uniprot:P17302"
    assert paper.source == PaperSource.UNIPROT
    assert "Gja1" in paper.abstract
    assert "382 aa" in paper.abstract
    assert "Rattus norvegicus" in paper.abstract
    assert "2ZW3" in paper.abstract
    assert "Cell membrane" in paper.abstract
    assert paper.publication_date is not None
    assert paper.publication_date.year == 2023


def test_uniprot_parse_entry_missing_accession_returns_none():
    collector = UniProtCollector.__new__(UniProtCollector)
    assert collector._parse_entry({}) is None


def test_uniprot_parse_entry_no_pdb_xref_notes_alphafold():
    collector = UniProtCollector.__new__(UniProtCollector)
    fixture = dict(UNIPROT_ENTRY_FIXTURE)
    fixture["uniProtKBCrossReferences"] = []
    paper = collector._parse_entry(fixture)
    assert "AlphaFold" in paper.abstract


# ======================================================================
# RCSB PDB
# ======================================================================
PDB_ENTRY_FIXTURE = {
    "struct": {"title": "Crystal structure of a gap junction channel"},
    "exptl": [{"method": "X-RAY DIFFRACTION"}],
    "rcsb_entry_info": {"resolution_combined": [3.5]},
    "struct_keywords": {"pdbx_keywords": "MEMBRANE PROTEIN"},
    "rcsb_accession_info": {"deposit_date": "2009-01-15T00:00:00Z"},
}


def test_pdb_parse_entry_basic_fields():
    collector = PDBCollector.__new__(PDBCollector)
    paper = collector._parse_entry("2ZW3", PDB_ENTRY_FIXTURE)

    assert paper is not None
    assert paper.paper_id == "pdb:2ZW3"
    assert paper.source == PaperSource.PDB
    assert "X-RAY DIFFRACTION" in paper.abstract
    assert "3.5 Angstrom" in paper.abstract
    assert paper.publication_date is not None
    assert paper.publication_date.year == 2009


def test_pdb_search_ids_parses_result_set(monkeypatch):
    collector = PDBCollector.__new__(PDBCollector)

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"result_set": [{"identifier": "2ZW3"}, {"identifier": "1TXN"}]}

    class FakeClient:
        def post(self, url, json, timeout):
            return FakeResponse()

    collector.client = FakeClient()
    ids = collector._search_ids("connexin", 10)
    assert ids == ["2ZW3", "1TXN"]


# ======================================================================
# AlphaFold DB
# ======================================================================
ALPHAFOLD_FIXTURE = [
    {
        "entryId": "AF-P17302-F1",
        "gene": "Gja1",
        "uniprotAccession": "P17302",
        "organismScientificName": "Rattus norvegicus",
        "modelCreatedDate": "2022-06-01",
        "latestVersion": 4,
        "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P17302-F1-model_v4.pdb",
        "globalMetricValue": 92.5,
    }
]


def test_alphafold_parse_prediction_basic_fields():
    collector = AlphaFoldCollector.__new__(AlphaFoldCollector)
    paper = collector._parse_prediction("P17302", ALPHAFOLD_FIXTURE)

    assert paper is not None
    assert paper.paper_id == "alphafold:AF-P17302-F1"
    assert paper.source == PaperSource.ALPHAFOLD
    assert "92.5" in paper.abstract
    assert "very high" in paper.abstract
    assert paper.publication_date.year == 2022


def test_alphafold_parse_prediction_empty_response():
    collector = AlphaFoldCollector.__new__(AlphaFoldCollector)
    assert collector._parse_prediction("Q00000", []) is None
    assert collector._parse_prediction("Q00000", None) is None


# ======================================================================
# NCBI Gene
# ======================================================================
NCBI_GENE_RECORD_FIXTURE = {
    "name": "GJA1",
    "description": "gap junction protein alpha 1",
    "organism": {"scientificname": "Homo sapiens"},
    "chromosome": "6",
    "maplocation": "6q22.31",
    "summary": "This gene encodes a member of the connexin gene family.",
    "otheraliases": "CX43, ODDD",
    "genetype": "protein-coding",
    "genomicinfo": [{"chraccver": "NC_000006.12"}],
}


def test_ncbi_gene_parse_record_basic_fields():
    collector = NCBIGeneCollector.__new__(NCBIGeneCollector)
    paper = collector._parse_record("2697", NCBI_GENE_RECORD_FIXTURE)

    assert paper is not None
    assert paper.paper_id == "ncbigene:2697"
    assert paper.source == PaperSource.NCBI_GENE
    assert "GJA1" in paper.title
    assert "chromosome 6" in paper.abstract
    assert "NC_000006.12" in paper.abstract
    assert "connexin gene family" in paper.abstract
    assert "CX43" in paper.keywords
