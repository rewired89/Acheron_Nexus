"""Turn ingested RAG chunks into structured, machine-usable parameter records.

This is the bridge between the RAG corpus (free text chunks in ChromaDB) and
any downstream simulation: instead of a paragraph of prose, a simulation
needs discrete records of the shape

    (subject) -[relationship_type]-> (object), rate/affinity: <value or UNKNOWN>

Confidence tiers come from `rag/science_filter.py`'s existing evidence-scoring
function (`score_evidence`) — this module does not implement a second scoring
system. It just buckets that same 0-1 composite score into HIGH/MEDIUM/LOW.

`science_filter._ORGANISM_TIERS` originally had no tier for bacteria (it was
built for planarian-vs-comparative-model scoring), which meant every
SubtiWiki (B. subtilis) chunk scored organism_match=0.0 no matter what
target_organism string was passed — Acheron Nexus has no use for a scorer
that can't recognize the one organism a whole collector exists to describe.
A "bacteria" tier was added to `_ORGANISM_TIERS` (keywords: bacillus,
b. subtilis, subtilis, bacteri, prokaryot, gram-positive/negative,
e. coli/escherichia coli) so B. subtilis text now scores organism_match=1.0
when `score_evidence()` is called with `target_organism="bacteria"`, exactly
the same way PlanMine (S. mediterranea) text scores 1.0 against the existing
"planarian" tier. This module resolves the right target_organism per chunk
via `_target_organism_for()` below. It is still a one-line addition to the
existing scorer, not a second scoring system, and every other caller of
`score_evidence()`/`score_organism_match()` is unaffected since its default
`target_organism="planarian"` behavior is unchanged.

Storage: parameter records are written as one JSON file per record under
`data_dir / "parameters"`, mirroring `rag/ledger.py`'s existing
file-per-entry, glob-scanned pattern (deliberately not mixed into ChromaDB's
text-chunk collection, per the task's storage requirement).

Extraction is source-aware:
  - SubtiWiki chunks: regex targets the exact structured lines
    `collectors/subtiwiki.py` writes ("Interacts with ...", "Regulated by
    ... (mechanism, mode): ...", "Acts as a regulator: regulates N gene(s)
    and M operon(s) ...").
  - PlanMine chunks: regex targets the exact structured lines
    `collectors/planmine.py` writes ("Homology annotations: ...",
    "Protein domain annotations: ...", "ION CHANNEL / GAP JUNCTION
    RELEVANT ...", RNAi differential-expression bullets).
  - Everything else (literature: PubMed/bioRxiv/arXiv/UniProt/PDB/...):
    a best-effort heuristic fallback looking for "X binds/interacts
    with/regulates/activates/inhibits Y" phrasing plus cited kinetic
    constants (Kd, Ki, kcat, EC50, IC50 with units). This is explicitly
    labeled lower-confidence: free text was never written to be machine
    parsed, unlike SubtiWiki/PlanMine's own structured summaries.

No numeric invention: `rate_or_affinity` is either a literal substring the
source text actually contains, or the literal string "UNKNOWN" — never
computed, converted, or guessed.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from acheron.config import get_settings
from acheron.models import QueryResult, TextChunk
from acheron.rag.science_filter import score_evidence

logger = logging.getLogger(__name__)

UNKNOWN = "UNKNOWN"


# ======================================================================
# Confidence tiers — thin wrapper around science_filter.score_evidence,
# not a second scoring system.
# ======================================================================
class ConfidenceTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


def _tier_from_score(total: float) -> ConfidenceTier:
    """Bucket science_filter's composite [0-1] score into a tier.

    science_filter's own weights sum to 1.0, so a perfect match across every
    sub-score tops out at total=1.0. These cut points are not a new scoring
    function, just a readable label over the existing number.
    """
    if total >= 0.7:
        return ConfidenceTier.HIGH
    if total >= 0.4:
        return ConfidenceTier.MEDIUM
    return ConfidenceTier.LOW


# ======================================================================
# Parameter record — output shape, kept local to this module (Phase 3
# only names this one new file; nothing added to models.py).
# ======================================================================
class ParameterRecord(BaseModel):
    """One structured, machine-usable (subject) -[relationship]-> (object) fact."""

    record_id: str
    subject: str
    relationship_type: str
    object: str
    rate_or_affinity: str = UNKNOWN
    evidence_text: str
    organism: str = ""
    source_type: str = ""
    source: str = ""
    paper_id: str = ""
    chunk_id: str = ""
    confidence_tier: ConfidenceTier = ConfidenceTier.LOW
    confidence_score: float = 0.0
    confidence_breakdown: dict = Field(default_factory=dict)
    heuristic: bool = False


# ======================================================================
# Storage — flat JSON files under data_dir/parameters, mirroring
# rag/ledger.py's ExperimentLedger pattern (file-per-record, glob-scanned).
# ======================================================================
class ParameterStore:
    """Persistent store for extracted ParameterRecords, separate from ChromaDB."""

    def __init__(self, params_dir: Optional[Path] = None) -> None:
        settings = get_settings()
        self.params_dir = params_dir or settings.data_dir / "parameters"
        self.params_dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: ParameterRecord) -> Path:
        dest = self.params_dir / f"{record.record_id}.json"
        dest.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return dest

    def save_all(self, records: list[ParameterRecord]) -> int:
        for record in records:
            self.save(record)
        return len(records)

    def list_records(self, organism: Optional[str] = None) -> list[ParameterRecord]:
        records = []
        for path in sorted(self.params_dir.glob("param-*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                record = ParameterRecord(**data)
            except Exception:
                logger.debug("Skipping malformed parameter record: %s", path.name)
                continue
            if organism and record.organism != organism:
                continue
            records.append(record)
        return records

    def count(self) -> int:
        return len(list(self.params_dir.glob("param-*.json")))


def _new_record_id() -> str:
    return f"param-{uuid.uuid4().hex[:12]}"


# ======================================================================
# Confidence scoring for a chunk (reuses science_filter.score_evidence)
# ======================================================================
# Maps a Paper/TextChunk's own `organism` string to the _ORGANISM_TIERS key
# that actually describes it, so score_evidence() gets asked about the
# organism a chunk is really about instead of always defaulting to
# "planarian". Extend this map, not science_filter's scorer, when a new
# curated-database organism is added.
_ORGANISM_TARGET_MAP = {
    "b. subtilis": "bacteria",
    "s. mediterranea": "planarian",
}


def _target_organism_for(organism: str) -> str:
    return _ORGANISM_TARGET_MAP.get((organism or "").lower(), organism or "planarian")


def _score_chunk(chunk: TextChunk) -> tuple[ConfidenceTier, float, dict]:
    meta = chunk.metadata or {}
    authors = meta.get("authors") or []
    if isinstance(authors, str):
        try:
            authors = json.loads(authors)
        except Exception:
            authors = []

    target_organism = _target_organism_for(chunk.organism)

    result = QueryResult(
        text=chunk.text,
        paper_id=chunk.paper_id,
        paper_title=meta.get("title", ""),
        authors=authors,
        doi=meta.get("doi") or None,
        pmid=meta.get("pmid") or None,
        pmcid=meta.get("pmcid") or None,
        section=chunk.section,
        source=_source_value(chunk),
        excerpt=chunk.excerpt,
    )
    score = score_evidence(result, target_organism=target_organism)
    breakdown = {
        "organism_match": score.organism_match,
        "primary_data": score.primary_data,
        "measurement_specificity": score.measurement_specificity,
        "recency": score.recency,
        "citation_quality": score.citation_quality,
        "total": score.total,
    }
    return _tier_from_score(score.total), score.total, breakdown


# ======================================================================
# Kinetic constant detection (used by the literature fallback only —
# SubtiWiki/PlanMine's own structured text never cites Kd/Ki/kcat/EC50/IC50)
# ======================================================================
_RATE_PATTERN = re.compile(
    r"\b(K[dDiI]|k[_ ]?cat|EC50|IC50|K[_ ]?m)\s*[=:≈]?\s*"
    r"\d+\.?\d*\s*(n|µ|u|m)?M\b",
    re.IGNORECASE,
)


def _find_rate_or_affinity(text: str) -> str:
    match = _RATE_PATTERN.search(text)
    return match.group(0) if match else UNKNOWN


# ======================================================================
# Subject resolution: prefer the chunk's own "Gene: X" line (SubtiWiki/
# PlanMine always emit this as their first line), fall back to the
# paper title.
# ======================================================================
_GENE_LINE = re.compile(r"^Gene:\s*(.+)$", re.MULTILINE)


def _resolve_subject(chunk: TextChunk) -> str:
    match = _GENE_LINE.search(chunk.text)
    if match:
        return match.group(1).strip()
    title = (chunk.metadata or {}).get("title", "")
    return title.split(" (")[0].strip() if title else chunk.paper_id


# ======================================================================
# Source-specific extraction: SubtiWiki
# ======================================================================
_SUBTIWIKI_INTERACTS = re.compile(r"^\s*-\s*Interacts with (.+?)(?:\s*\((.*)\))?\s*$", re.MULTILINE)
_SUBTIWIKI_REGULATED_BY = re.compile(r"^\s*-\s*Regulated by (.+?)(?:\s*\((.*?)\))?(?::\s*(.*))?\s*$", re.MULTILINE)
_SUBTIWIKI_REGULON_SUMMARY = re.compile(
    r"Acts as a regulator: regulates (\d+) gene\(s\) and (\d+) operon\(s\) directly"
)


def _extract_subtiwiki(chunk: TextChunk, subject: str) -> list[tuple[str, str, str, str]]:
    """Returns (relationship_type, object, rate_or_affinity, evidence_text) tuples."""
    records: list[tuple[str, str, str, str]] = []

    for match in _SUBTIWIKI_INTERACTS.finditer(chunk.text):
        partners_raw = match.group(1)
        for partner in [p.strip() for p in partners_raw.split(",") if p.strip()]:
            records.append(("interacts_with", partner, UNKNOWN, match.group(0).strip()))

    for match in _SUBTIWIKI_REGULATED_BY.finditer(chunk.text):
        regulator = match.group(1).strip()
        records.append(("regulated_by", regulator, UNKNOWN, match.group(0).strip()))

    summary = _SUBTIWIKI_REGULON_SUMMARY.search(chunk.text)
    if summary:
        n_genes, n_operons = summary.group(1), summary.group(2)
        records.append((
            "regulates",
            f"{n_genes} gene(s), {n_operons} operon(s) (regulon summary, not individual targets)",
            UNKNOWN,
            summary.group(0).strip(),
        ))

    return records


# ======================================================================
# Source-specific extraction: PlanMine
# ======================================================================
_PLANMINE_HOMOLOGY = re.compile(r"Homology annotations:\s*(.+)")
_PLANMINE_ION_CHANNEL = re.compile(r"ION CHANNEL / GAP JUNCTION RELEVANT \(matched domain annotation\):\s*(.+)")
_PLANMINE_DOMAINS = re.compile(r"^Protein domain annotations:\s*(.+)$", re.MULTILINE)
_PLANMINE_RNAI_HEADER = re.compile(r"RNAi/RNA-seq differential-expression results.*:\n")


def _extract_planmine(chunk: TextChunk, subject: str) -> list[tuple[str, str, str, str]]:
    records: list[tuple[str, str, str, str]] = []

    homology = _PLANMINE_HOMOLOGY.search(chunk.text)
    if homology:
        for item in [i.strip() for i in homology.group(1).split(";") if i.strip() and not i.strip().startswith("(")]:
            m = re.match(r"(\S+)\s*\((\w+)\)", item)
            target = m.group(1) if m else item
            htype = m.group(2) if m else ""
            records.append(("homologous_to", target, UNKNOWN, f"{item} [{htype}]" if htype else item))

    ion_channel = _PLANMINE_ION_CHANNEL.search(chunk.text)
    ion_channel_labels: set[str] = set()
    if ion_channel:
        for label in [l.strip() for l in ion_channel.group(1).split(";") if l.strip()]:
            ion_channel_labels.add(label)
            records.append(("has_domain_annotation", label, UNKNOWN, ion_channel.group(0).strip()))

    domains = _PLANMINE_DOMAINS.search(chunk.text)
    if domains:
        for label in [l.strip() for l in domains.group(1).split(";") if l.strip()]:
            if label in ion_channel_labels:
                continue  # already recorded via the ion-channel-flagged line, avoid duplicating
            records.append(("has_domain", label, UNKNOWN, domains.group(0).strip()))

    header_match = _PLANMINE_RNAI_HEADER.search(chunk.text)
    if header_match:
        rnai_section = chunk.text[header_match.end():]
        # Bulleted lines stop at the next unindented section header or end of text.
        for line in rnai_section.splitlines():
            if not line.strip():
                continue
            if not line.startswith("  -") and not line.startswith("  ("):
                break
            if line.startswith("  ("):
                continue
            content = line.strip().lstrip("- ").strip()
            score_match = re.search(r"score=(-?\d+\.?\d*)", content)
            records.append((
                "differential_expression_under",
                content,
                UNKNOWN,  # an RNAi differential-expression score is not a rate/affinity constant
                f"score={score_match.group(1)}" if score_match else content,
            ))

    return records


# ======================================================================
# Literature fallback — heuristic, explicitly lower confidence.
# ======================================================================
_LIT_PATTERNS = [
    (re.compile(r"\b([A-Za-z0-9\-]+)\s+binds\s+(?:to\s+)?([A-Za-z0-9\-]+)", re.IGNORECASE), "binds"),
    (re.compile(r"\b([A-Za-z0-9\-]+)\s+interacts with\s+([A-Za-z0-9\-]+)", re.IGNORECASE), "interacts_with"),
    (re.compile(r"\b([A-Za-z0-9\-]+)\s+regulates\s+([A-Za-z0-9\-]+)", re.IGNORECASE), "regulates"),
    (re.compile(r"\b([A-Za-z0-9\-]+)\s+activates\s+([A-Za-z0-9\-]+)", re.IGNORECASE), "activates"),
    (re.compile(r"\b([A-Za-z0-9\-]+)\s+inhibits\s+([A-Za-z0-9\-]+)", re.IGNORECASE), "inhibits"),
]


def _extract_literature(chunk: TextChunk, subject: str) -> list[tuple[str, str, str, str]]:
    records: list[tuple[str, str, str, str]] = []
    for pattern, relationship_type in _LIT_PATTERNS:
        for match in pattern.finditer(chunk.text):
            sentence_start = chunk.text.rfind(".", 0, match.start()) + 1
            sentence_end = chunk.text.find(".", match.end())
            sentence = chunk.text[sentence_start : sentence_end if sentence_end != -1 else None].strip()

            # A cited rate/affinity constant is often in the *next* sentence
            # rather than the exact one asserting the relationship, so widen
            # the search window by one more sentence -- still only ever
            # returning a literal substring the source actually contains,
            # never a computed or guessed value.
            window_end = sentence_end
            if sentence_end != -1:
                next_end = chunk.text.find(".", sentence_end + 1)
                window_end = next_end if next_end != -1 else len(chunk.text)
            window = chunk.text[sentence_start : window_end if window_end != -1 else None]

            records.append((
                relationship_type,
                match.group(2),
                _find_rate_or_affinity(window),
                sentence or match.group(0),
            ))
    return records


# ======================================================================
# Public entry points
# ======================================================================
def _source_value(chunk: TextChunk) -> str:
    """Normalize chunk.metadata["source"] to its plain string value.

    It may be a PaperSource enum instance (in-memory, straight from
    chunker.py) or a plain string (round-tripped through ChromaDB, which
    only stores plain strings) -- str(PaperSource.SUBTIWIKI) is
    "PaperSource.SUBTIWIKI", not "subtiwiki", so .value must be preferred
    when present.
    """
    raw = (chunk.metadata or {}).get("source", "")
    return str(getattr(raw, "value", raw))


def extract_from_chunk(chunk: TextChunk) -> list[ParameterRecord]:
    """Extract structured ParameterRecords from a single TextChunk.

    Dispatches by the chunk's source: SubtiWiki- and PlanMine-specific
    parsers target those collectors' own known text layout; every other
    source falls back to a heuristic literature parser.
    """
    source = _source_value(chunk)
    subject = _resolve_subject(chunk)
    is_heuristic = source not in ("subtiwiki", "planmine")

    if source == "subtiwiki":
        raw = _extract_subtiwiki(chunk, subject)
    elif source == "planmine":
        raw = _extract_planmine(chunk, subject)
    else:
        raw = _extract_literature(chunk, subject)

    if not raw:
        return []

    tier, total, breakdown = _score_chunk(chunk)

    records = []
    for relationship_type, obj, rate_or_affinity, evidence_text in raw:
        records.append(ParameterRecord(
            record_id=_new_record_id(),
            subject=subject,
            relationship_type=relationship_type,
            object=obj,
            rate_or_affinity=rate_or_affinity,
            evidence_text=evidence_text,
            organism=chunk.organism or "",
            source_type=str(chunk.source_type or ""),
            source=source,
            paper_id=chunk.paper_id,
            chunk_id=chunk.chunk_id,
            confidence_tier=tier,
            confidence_score=total,
            confidence_breakdown=breakdown,
            heuristic=is_heuristic,
        ))
    return records


def _reconstruct_chunk(chunk_id: str, document: str, meta: dict) -> TextChunk:
    """Rebuild a minimal TextChunk from what's actually stored in ChromaDB
    (see vectorstore/store.py's add_chunks() for the exact metadata shape).
    """
    return TextChunk(
        chunk_id=chunk_id,
        paper_id=meta.get("paper_id", ""),
        text=document,
        section=meta.get("section", ""),
        chunk_index=meta.get("chunk_index", 0) or 0,
        metadata={
            "title": meta.get("title", ""),
            "authors": meta.get("authors", "[]"),
            "doi": meta.get("doi", ""),
            "pmid": meta.get("pmid", ""),
            "pmcid": meta.get("pmcid", ""),
            "source": meta.get("source", ""),
            "date": meta.get("date", ""),
            "url": meta.get("url", ""),
        },
        source_file=meta.get("source_file", ""),
        span_start=meta.get("span_start", 0) or 0,
        span_end=meta.get("span_end", 0) or 0,
        excerpt=meta.get("excerpt", ""),
        xpath=meta.get("xpath", ""),
        organism=meta.get("organism", "") or "",
        source_type=meta.get("source_type") or None,
    )


def extract_from_store(store, params_dir: Optional[Path] = None, persist: bool = True) -> list[ParameterRecord]:
    """Pull every chunk currently indexed in `store` and extract parameter
    records from each. Persists them to ParameterStore unless persist=False.

    `store` is an already-constructed `vectorstore.store.VectorStore`. Kept
    untyped here to avoid importing chromadb at module load for callers that
    only need extract_from_chunk (e.g. unit tests).
    """
    raw = store.collection.get(include=["documents", "metadatas"])
    ids = raw.get("ids", [])
    documents = raw.get("documents", [])
    metadatas = raw.get("metadatas", [])

    all_records: list[ParameterRecord] = []
    param_store = ParameterStore(params_dir) if persist else None

    for chunk_id, document, meta in zip(ids, documents, metadatas):
        chunk = _reconstruct_chunk(chunk_id, document, meta or {})
        records = extract_from_chunk(chunk)
        all_records.extend(records)
        if param_store:
            param_store.save_all(records)

    logger.info("Extracted %d parameter records from %d indexed chunks", len(all_records), len(ids))
    return all_records
