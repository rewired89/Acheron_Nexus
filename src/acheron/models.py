"""Core domain models used across the application."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ======================================================================
# Enumerations
# ======================================================================
class PaperSource(str, Enum):
    PUBMED = "pubmed"
    BIORXIV = "biorxiv"
    ARXIV = "arxiv"
    PHYSIONET = "physionet"
    MANUAL = "manual"
    UNIPROT = "uniprot"
    PDB = "pdb"
    ALPHAFOLD = "alphafold"
    NCBI_GENE = "ncbi_gene"


class EpistemicTag(str, Enum):
    """Classification for separating evidence, inference, and speculation."""

    EVIDENCE = "evidence"
    INFERENCE = "inference"
    SPECULATION = "speculation"


class ClaimStatus(str, Enum):
    """Verification status for an evidence claim."""

    SUPPORTED = "supported"
    MIXED = "mixed"
    UNCLEAR = "unclear"
    UNSUPPORTED = "unsupported"


class NexusMode(str, Enum):
    """Operating mode for the hypothesis engine."""

    EVIDENCE = "evidence"  # MODE 1: evidence-grounded summary
    HYPOTHESIS = "hypothesis"  # MODE 2: IBE hypothesis generation
    SYNTHESIS = "synthesis"  # MODE 3: systems synthesis / design
    DECISION = "decision"  # MODE 4: engineering verdict (verdict-first)
    TUTOR = "tutor"  # MODE 5: educational explanation with glossary/analogies


# ======================================================================
# Paper models
# ======================================================================
class SourceProvenance(BaseModel):
    """Provenance tracking for verification of ingested records."""

    provider: str = ""  # e.g., "NCBI"
    database: str = ""  # e.g., "pubmed", "pmc"
    fetched_at_utc: str = ""
    request_query: str = ""
    url_used: str = ""


class Paper(BaseModel):
    """Represents a single research paper with full metadata."""

    paper_id: str = Field(..., description="Unique identifier (DOI or source-specific ID)")
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    publication_date: Optional[date] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    arxiv_id: Optional[str] = None
    source: PaperSource = PaperSource.MANUAL
    journal: str = ""
    keywords: list[str] = Field(default_factory=list)
    url: str = ""
    pdf_path: Optional[str] = None
    full_text: Optional[str] = None
    sections: list[PaperSection] = Field(default_factory=list)
    tables: list[TableData] = Field(default_factory=list)
    citation_count: Optional[int] = None
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    provenance: Optional[SourceProvenance] = None

    def display_citation(self) -> str:
        """Return a human-readable citation string."""
        author_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            author_str += " et al."
        year = self.publication_date.year if self.publication_date else "n.d."
        doi_str = f" DOI: {self.doi}" if self.doi else ""
        return f"{author_str} ({year}). {self.title}. {self.journal}.{doi_str}"


class PaperSection(BaseModel):
    """A section from a parsed paper."""

    heading: str = ""
    text: str = ""
    order: int = 0


class TableData(BaseModel):
    """A table extracted from a paper."""

    caption: str = ""
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)

    def to_text(self) -> str:
        """Render the table as a readable text block."""
        lines = []
        if self.caption:
            lines.append(f"Table: {self.caption}")
        if self.headers:
            lines.append(" | ".join(self.headers))
            lines.append("-" * (len(" | ".join(self.headers))))
        for row in self.rows:
            lines.append(" | ".join(row))
        return "\n".join(lines)


class TextChunk(BaseModel):
    """A chunk of text ready for embedding and storage.

    Includes evidence span tracking for precise citations.
    """

    chunk_id: str
    paper_id: str
    text: str
    section: str = ""
    chunk_index: int = 0
    metadata: dict = Field(default_factory=dict)
    # Evidence span fields for precise citations
    source_file: str = ""
    span_start: int = 0  # byte/char offset in source
    span_end: int = 0
    excerpt: str = ""  # short excerpt (200-400 chars) for display
    xpath: str = ""  # location in structured document (e.g., NXML)


# ======================================================================
# Query and response models
# ======================================================================
class QueryResult(BaseModel):
    """A single retrieval result with source attribution.

    Includes evidence span for precise citations: file, offsets, excerpt.
    """

    text: str
    paper_id: str
    paper_title: str
    authors: list[str] = Field(default_factory=list)
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    section: str = ""
    relevance_score: float = 0.0
    url: str = Field(default="", description="Direct link to the source record (paper, protein, structure, or gene entry)")
    source: str = Field(default="", description="Origin collector, e.g. pubmed, uniprot, pdb, alphafold, ncbi_gene")
    # Evidence span fields
    source_file: str = ""
    span_start: int = 0
    span_end: int = 0
    excerpt: str = ""  # short displayable excerpt
    xpath: str = ""  # location in structured doc

    def format_citation(self) -> str:
        author_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            author_str += " et al."
        doi_part = f" (DOI: {self.doi})" if self.doi else ""
        pmid_part = f" PMID:{self.pmid}" if self.pmid else ""
        return f"[{author_str}. \"{self.paper_title}\"{doi_part}{pmid_part}]"

    def resolved_link(self) -> str:
        """Best available direct link to the original record, for readers who
        want to verify a claim against the primary source themselves.

        Preference order: DOI (most stable) > PMID (PubMed) > stored URL
        (covers UniProt/PDB/AlphaFold/NCBI Gene entry pages and PDFs).
        """
        if self.doi:
            return f"https://doi.org/{self.doi}"
        if self.pmid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"
        return self.url or ""

    def format_evidence_span(self) -> str:
        """Format evidence span for display."""
        loc = self.source_file or self.paper_id
        if self.xpath:
            loc = f"{loc}#{self.xpath}"
        elif self.span_start or self.span_end:
            loc = f"{loc}:{self.span_start}-{self.span_end}"
        excerpt = self.excerpt or self.text[:200]
        return f"{loc}\n  \"{excerpt}...\""


class StructuredVariable(BaseModel):
    """A variable extracted from source material during the discovery loop.

    Covers the core bioelectric variables: Vmem (membrane voltage),
    EF (endogenous electric fields), Gj (gap junctional conductance),
    ion channel types (K+, Na+, Ca2+, Cl-), perturbations, and outcomes.
    """

    name: str
    value: str = ""
    unit: str = ""
    context: str = ""
    source_ref: str = ""
    variable_type: str = ""  # vmem, ef, gj, ion_channel, perturbation, outcome, other


class Hypothesis(BaseModel):
    """A testable hypothesis generated from pattern comparison.

    Includes prior confidence (based on evidence density), predicted impact
    (what changes if true), and clear assumptions.
    """

    statement: str
    supporting_refs: list[str] = Field(default_factory=list)
    confidence: str = "low"  # low, medium, high — based on evidence density
    predicted_impact: str = ""  # what changes in our understanding if this is true
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions underlying this hypothesis",
    )
    validation_strategy: str = ""


class DiscoveryResult(BaseModel):
    """Structured output from the discovery loop.

    Separates evidence (from sources), inference (logical derivation),
    and speculation (hypotheses beyond direct evidence). Includes
    bioelectric schematic and validation path as mandatory sections.
    """

    query: str
    evidence: list[str] = Field(
        default_factory=list,
        description="Statements directly supported by retrieved sources",
    )
    inference: list[str] = Field(
        default_factory=list,
        description="Logical derivations from the evidence",
    )
    speculation: list[str] = Field(
        default_factory=list,
        description="Hypotheses that go beyond what the evidence directly states",
    )
    variables: list[StructuredVariable] = Field(
        default_factory=list,
        description="Structured variables extracted from sources",
    )
    hypotheses: list[Hypothesis] = Field(
        default_factory=list,
        description="Testable hypotheses generated from pattern comparison",
    )
    bioelectric_schematic: str = Field(
        default="",
        description="Hypothesized bioelectric circuit description, e.g. "
        "'Hyperpolarization of tissue X alters Gj, leading to "
        "suppression of pathway Y and morphological outcome Z'",
    )
    validation_path: list[str] = Field(
        default_factory=list,
        description="Proposed experimental or computational validation steps",
    )
    cross_species_notes: list[str] = Field(
        default_factory=list,
        description="Cross-species reasoning (Planaria <-> Xenopus <-> Mammalian)",
    )
    sources: list[QueryResult] = Field(default_factory=list)
    model_used: str = ""
    total_chunks_searched: int = 0
    uncertainty_notes: list[str] = Field(
        default_factory=list,
        description="Explicit declarations of uncertainty or missing data",
    )
    # Auto-routing fields
    detected_mode: str = Field(
        default="discovery",
        description="Mode that was actually used (discovery, decision, tutor, etc.)",
    )
    raw_output: str = Field(
        default="",
        description="Raw LLM output when auto-routed to a different mode",
    )


class RAGResponse(BaseModel):
    """Complete response from the RAG pipeline."""

    query: str
    answer: str
    sources: list[QueryResult] = Field(default_factory=list)
    model_used: str = ""
    total_chunks_searched: int = 0
    # Optional structured layers — populated when discovery mode is used
    evidence_statements: list[str] = Field(default_factory=list)
    inference_statements: list[str] = Field(default_factory=list)
    speculation_statements: list[str] = Field(default_factory=list)
    bioelectric_schematic: str = ""

    def format_full(self) -> str:
        """Format the response with inline citations for display."""
        lines = [self.answer, "", "--- Sources ---"]
        seen = set()
        for i, src in enumerate(self.sources, 1):
            key = src.paper_id
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  [{i}] {src.format_citation()}")
        return "\n".join(lines)


# ======================================================================
# Experiment ledger
# ======================================================================
class LedgerEntry(BaseModel):
    """A single entry in the experiment ledger."""

    entry_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    query: str
    evidence_summary: str = ""
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    variables: list[StructuredVariable] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    notes: str = ""
    tags: list[str] = Field(default_factory=list)


# ======================================================================
# Evidence Graph (Knowledge Graph for claim verification)
# ======================================================================
class EvidenceClaim(BaseModel):
    """An atomic claim extracted from source material (subject–predicate–object)."""

    claim_id: str = ""
    subject: str = ""
    predicate: str = ""
    object: str = ""
    source_refs: list[str] = Field(default_factory=list)  # [1], [2], etc.
    source_pmids: list[str] = Field(default_factory=list)
    support_count: int = 0
    contradiction_count: int = 0
    agreement_score: float = 0.0  # 0.0–1.0
    status: ClaimStatus = ClaimStatus.UNCLEAR
    recency: str = ""  # year of most recent source
    study_types: list[str] = Field(default_factory=list)  # review, primary, preprint


class EvidenceEdge(BaseModel):
    """A relationship between two claims in the evidence graph."""

    source_claim: str = ""  # claim_id
    target_claim: str = ""  # claim_id
    relation: str = ""  # supports, contradicts, depends-on


class EvidenceGraph(BaseModel):
    """Lightweight evidence graph for structured claim tracking.

    JSON-serializable so it can be logged to the ledger and reused across sessions.
    """

    claims: list[EvidenceClaim] = Field(default_factory=list)
    edges: list[EvidenceEdge] = Field(default_factory=list)
    query: str = ""
    timestamp: str = ""

    def supported_claims(self) -> list[EvidenceClaim]:
        return [c for c in self.claims if c.status == ClaimStatus.SUPPORTED]

    def contested_claims(self) -> list[EvidenceClaim]:
        return [c for c in self.claims if c.status == ClaimStatus.MIXED]

    def to_summary(self) -> str:
        """Human-readable summary of the evidence graph."""
        lines = [f"Evidence Graph ({len(self.claims)} claims, {len(self.edges)} edges)"]
        for c in self.claims:
            lines.append(
                f"  [{c.status.value}] {c.subject} {c.predicate} {c.object} "
                f"(agreement={c.agreement_score:.2f}, refs={c.source_refs})"
            )
        return "\n".join(lines)


# ======================================================================
# Ranked Hypothesis (IBE: Inference to the Best Explanation)
# ======================================================================
class RankedHypothesis(BaseModel):
    """A hypothesis ranked using Inference to the Best Explanation (IBE).

    Includes falsification-first output (Popper-style) and uncertainty calibration.
    """

    hypothesis_id: str = ""
    statement: str = ""
    rank: int = 0
    # IBE scoring dimensions
    explanatory_power: float = 0.0  # 0–1: how many supported claims does it explain
    simplicity: float = 0.0  # 0–1: fewest extra assumptions
    consistency: float = 0.0  # 0–1: avoids contradicting strong evidence
    mechanistic_plausibility: float = 0.0  # 0–1: fits known biology/physics
    overall_score: float = 0.0
    rationale: str = ""
    # Falsification-first output
    predictions: list[str] = Field(
        default_factory=list,
        description="What would we expect to observe if true?",
    )
    falsifiers: list[str] = Field(
        default_factory=list,
        description="What result would falsify this?",
    )
    minimal_test: str = Field(
        default="",
        description="What minimal experiment or observation could test this?",
    )
    # Uncertainty calibration
    confidence: int = Field(default=0, ge=0, le=100, description="Confidence score 0–100")
    confidence_justification: str = ""
    assumptions: list[str] = Field(default_factory=list)
    known_unknowns: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(
        default_factory=list,
        description="Most likely failure modes of this hypothesis",
    )
    supporting_refs: list[str] = Field(default_factory=list)


class HypothesisEngineResult(BaseModel):
    """Full output from the Evidence-Bound Hypothesis Engine."""

    query: str = ""
    mode: NexusMode = NexusMode.EVIDENCE
    evidence_graph: EvidenceGraph = Field(default_factory=EvidenceGraph)
    hypotheses: list[RankedHypothesis] = Field(default_factory=list)
    next_queries: list[str] = Field(
        default_factory=list,
        description="Recommended search queries for additional evidence",
    )
    confidence: int = Field(default=0, description="Overall confidence 0–100")
    confidence_justification: str = ""
    uncertainty_notes: list[str] = Field(default_factory=list)
    sources: list[QueryResult] = Field(default_factory=list)
    model_used: str = ""
    total_chunks_searched: int = 0
    live_sources_fetched: int = 0
    # Raw LLM output - especially important for Decision/Calculation mode
    raw_output: str = Field(default="", description="Raw LLM response text")


# Forward-reference resolution
Paper.model_rebuild()
