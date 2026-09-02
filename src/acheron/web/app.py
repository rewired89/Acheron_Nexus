"""FastAPI web application for Acheron Nexus — bioelectric research intelligence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from acheron.config import get_settings
from acheron.extraction.chunker import TextChunker
from acheron.extraction.pdf_parser import PDFParser
from acheron.models import PaperSource
from acheron.rag.pipeline import RAGPipeline
from acheron.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)

app = FastAPI(title="Acheron Nexus", version="0.2.0")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Lazy singletons
_store: VectorStore | None = None
_pipeline: RAGPipeline | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(store=get_store())
    return _pipeline


def _source_dict(s) -> dict:
    """Serialize a QueryResult for the API, including a direct link so a
    reader can jump straight to the original paper / protein / structure /
    gene record — not just the excerpt Nexus retrieved."""
    return {
        "text": s.text[:500],
        "paper_title": s.paper_title,
        "authors": s.authors,
        "doi": s.doi,
        "pmid": s.pmid,
        "section": s.section,
        "score": s.relevance_score,
        "source": s.source,
        "link": s.resolved_link(),
    }


# ======================================================================
# API models
# ======================================================================
class QueryRequest(BaseModel):
    question: str
    source_filter: Optional[str] = None
    n_results: int = 10
    retrieve_only: bool = False
    discover_mode: bool = False
    live: bool = False
    mode: Optional[str] = None  # evidence, hypothesis, synthesis, or None for auto


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    model_used: str
    total_chunks_searched: int
    plain_summary: str = ""
    evidence: list[str] = []
    inference: list[str] = []
    speculation: list[str] = []
    bioelectric_schematic: str = ""


class DiscoverResponse(BaseModel):
    evidence: list[str]
    inference: list[str]
    speculation: list[str]
    variables: list[dict]
    hypotheses: list[dict]
    bioelectric_schematic: str = ""
    validation_path: list[str] = []
    cross_species_notes: list[str] = []
    uncertainty: list[str]
    sources: list[dict]
    model_used: str
    total_chunks_searched: int
    # New fields for auto-routed Decision/Calculation answers
    raw_output: str = ""
    detected_mode: str = "discovery"


class StatsResponse(BaseModel):
    total_papers: int
    total_chunks: int
    papers_by_source: dict[str, int]
    ledger_entries: int = 0


# ======================================================================
# Web UI
# ======================================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    store = get_store()
    stats = {
        "total_papers": len(store.list_papers()),
        "total_chunks": store.count(),
    }
    return templates.TemplateResponse(request, "index.html", {"stats": stats})


# ======================================================================
# API endpoints
# ======================================================================
def _compute_error_json() -> JSONResponse:
    """Return a 503 JSON error when Compute layer is unavailable."""
    settings = get_settings()
    provider = settings.llm_provider
    if provider == "anthropic":
        key_hint = "Set ANTHROPIC_API_KEY in your .env file."
    else:
        key_hint = "Set ACHERON_LLM_API_KEY or OPENAI_API_KEY in your .env file."
    return JSONResponse(
        status_code=503,
        content={
            "error": "Compute layer unavailable",
            "provider": provider,
            "detail": f"Provider '{provider}' is not configured. {key_hint}",
            "retrieval_available": True,
            "hint": "Use retrieve_only=true or 'acheron query -r' for retrieval-only mode.",
        },
    )


@app.post("/api/query", response_model=QueryResponse)
async def api_query(req: QueryRequest):
    pipeline = get_pipeline()

    if req.retrieve_only:
        results = pipeline.retrieve_only(
            req.question, n_results=req.n_results, filter_source=req.source_filter
        )
        return QueryResponse(
            answer="",
            sources=[_source_dict(r) for r in results],
            model_used="",
            total_chunks_searched=len(results),
        )

    settings = get_settings()
    if not settings.compute_available:
        return _compute_error_json()

    response = pipeline.query(
        req.question, filter_source=req.source_filter, n_results=req.n_results
    )
    return QueryResponse(
        answer=response.answer,
        sources=[_source_dict(s) for s in response.sources],
        model_used=response.model_used,
        total_chunks_searched=response.total_chunks_searched,
        plain_summary=response.plain_summary,
        evidence=response.evidence_statements,
        inference=response.inference_statements,
        speculation=response.speculation_statements,
        bioelectric_schematic=response.bioelectric_schematic,
    )


@app.post("/api/discover", response_model=DiscoverResponse)
async def api_discover(req: QueryRequest):
    """Execute the discovery loop and return structured results."""
    from acheron.rag.ledger import ExperimentLedger

    settings = get_settings()
    if not settings.compute_available:
        return _compute_error_json()

    pipeline = get_pipeline()
    result = pipeline.discover(
        req.question, filter_source=req.source_filter, n_results=req.n_results
    )

    # Auto-log to ledger
    ledger = ExperimentLedger()
    ledger.record(result)

    return DiscoverResponse(
        evidence=result.evidence,
        inference=result.inference,
        speculation=result.speculation,
        variables=[
            {
                "name": v.name,
                "value": v.value,
                "unit": v.unit,
                "type": v.variable_type,
                "source_ref": v.source_ref,
            }
            for v in result.variables
        ],
        hypotheses=[
            {
                "statement": h.statement,
                "confidence": h.confidence,
                "predicted_impact": h.predicted_impact,
                "assumptions": h.assumptions,
                "refs": h.supporting_refs,
                "validation": h.validation_strategy,
            }
            for h in result.hypotheses
        ],
        bioelectric_schematic=result.bioelectric_schematic,
        validation_path=result.validation_path,
        cross_species_notes=result.cross_species_notes,
        uncertainty=result.uncertainty_notes,
        sources=[_source_dict(s) for s in result.sources],
        model_used=result.model_used,
        total_chunks_searched=result.total_chunks_searched,
        # Include the actual answer for auto-routed queries
        raw_output=getattr(result, 'raw_output', ''),
        detected_mode=getattr(result, 'detected_mode', 'discovery'),
    )


@app.post("/api/analyze")
async def api_analyze(req: QueryRequest):
    """Evidence-Bound Hypothesis Engine: evidence graph + IBE hypotheses + falsification."""
    settings = get_settings()
    if not settings.compute_available:
        return _compute_error_json()

    pipeline = get_pipeline()
    result = pipeline.analyze(
        req.question,
        mode=req.mode,
        live=req.live,
        filter_source=req.source_filter,
        n_results=req.n_results,
    )
    return {
        "query": result.query,
        "mode": result.mode.value,
        "evidence_graph": {
            "claims": [c.model_dump() for c in result.evidence_graph.claims],
            "edges": [e.model_dump() for e in result.evidence_graph.edges],
        },
        "hypotheses": [
            {
                "id": h.hypothesis_id,
                "rank": h.rank,
                "statement": h.statement,
                "overall_score": h.overall_score,
                "explanatory_power": h.explanatory_power,
                "simplicity": h.simplicity,
                "consistency": h.consistency,
                "mechanistic_plausibility": h.mechanistic_plausibility,
                "rationale": h.rationale,
                "predictions": h.predictions,
                "falsifiers": h.falsifiers,
                "minimal_test": h.minimal_test,
                "confidence": h.confidence,
                "confidence_justification": h.confidence_justification,
                "assumptions": h.assumptions,
                "known_unknowns": h.known_unknowns,
                "failure_modes": h.failure_modes,
                "refs": h.supporting_refs,
            }
            for h in result.hypotheses
        ],
        "next_queries": result.next_queries,
        "confidence": result.confidence,
        "confidence_justification": result.confidence_justification,
        "uncertainty": result.uncertainty_notes,
        "sources": [_source_dict(s) for s in result.sources],
        "model_used": result.model_used,
        "total_chunks_searched": result.total_chunks_searched,
        "live_sources_fetched": result.live_sources_fetched,
    }


@app.get("/api/stats", response_model=StatsResponse)
async def api_stats():
    from acheron.rag.ledger import ExperimentLedger

    store = get_store()
    papers = store.list_papers()
    by_source: dict[str, int] = {}
    for p in papers:
        s = p.get("source", "unknown")
        by_source[s] = by_source.get(s, 0) + 1

    ledger = ExperimentLedger()

    return StatsResponse(
        total_papers=len(papers),
        total_chunks=store.count(),
        papers_by_source=by_source,
        ledger_entries=ledger.count(),
    )


@app.get("/api/papers")
async def api_papers():
    store = get_store()
    return store.list_papers()


@app.get("/api/ledger")
async def api_ledger():
    """Return all experiment ledger entries."""
    from acheron.rag.ledger import ExperimentLedger

    ledger = ExperimentLedger()
    entries = ledger.list_entries()
    return [e.model_dump(mode="json") for e in entries]


# ======================================================================
# FAST ENDPOINT - Direct LLM call with minimal prompt for <1min responses
# ======================================================================
@app.post("/api/fast")
async def api_fast(req: QueryRequest):
    """Fast decision/calculation endpoint - bypasses complex pipeline.

    Use this for quick answers to calculation or viability questions.
    Returns the raw LLM response with minimal processing.
    """
    from acheron.rag.hypothesis_engine import FAST_DECISION_PROMPT, FAST_QUERY_TEMPLATE

    settings = get_settings()
    if not settings.compute_available:
        return _compute_error_json()

    pipeline = get_pipeline()

    # Retrieve minimal context (just 4 chunks for speed)
    results = pipeline.store.search(
        query=req.question,
        n_results=4,
        filter_source=req.source_filter
    )

    # Format context
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"[{i}] {r.paper_title}: {r.text[:300]}...")
    context = "\n\n".join(context_parts)

    # Build the prompt
    user_prompt = FAST_QUERY_TEMPLATE.format(context=context, query=req.question)

    # Direct LLM call
    raw_output = pipeline._generate_with_system(
        system_prompt=FAST_DECISION_PROMPT,
        user_prompt=user_prompt,
        max_tokens=1000,  # Limit response length for speed
    )

    return {
        "question": req.question,
        "answer": raw_output,
        "sources": [
            {"title": r.paper_title, "doi": r.doi, "link": r.resolved_link(), "source": r.source}
            for r in results
        ],
        "mode": "fast",
    }


@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    title: str = Form(""),
    doi: str = Form(""),
    authors: str = Form(""),
):
    """Upload and index a PDF into the Library."""
    settings = get_settings()
    pdf_dir = settings.pdf_dir
    pdf_dir.mkdir(parents=True, exist_ok=True)

    dest = pdf_dir / file.filename
    content = await file.read()
    dest.write_bytes(content)

    parser = PDFParser()
    chunker = TextChunker()
    store = get_store()

    paper = parser.parse(str(dest))
    if not paper:
        return JSONResponse({"error": "Failed to parse PDF"}, status_code=400)

    if title:
        paper.title = title
    if doi:
        paper.doi = doi
        paper.paper_id = doi
    if authors:
        paper.authors = [a.strip() for a in authors.split(",")]
    paper.source = PaperSource.MANUAL

    chunks = chunker.chunk_paper(paper)
    added = store.add_chunks(chunks)

    return {"message": f"Indexed {added} chunks", "title": paper.title, "total": store.count()}
