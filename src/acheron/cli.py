"""Acheron Nexus CLI — System-2 bioelectric research engine.

Three-layer architecture:
  Layer 1 (Knowledge)  — immutable source corpus
  Layer 2 (Synthesis)  — semantic retrieval, bioelectric variable extraction
  Layer 3 (Discovery)  — analysis, pattern detection, hypothesis generation,
                          bioelectric schematic construction, validation paths
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from acheron.config import get_settings

console = Console()

# Default search topics for bioelectricity research (literature sources)
DEFAULT_TOPICS = {
    "planarian_bioelectricity": (
        "planarian bioelectricity membrane voltage regeneration"
    ),
    "eeg_cognitive": (
        "EEG cognitive patterns brain oscillations neural dynamics"
    ),
    "ion_channels": (
        "ion channel dynamics voltage-gated signaling electrophysiology"
    ),
    "bioelectric_morphogenesis": (
        "bioelectric morphogenesis pattern formation developmental bioelectricity"
    ),
    "regeneration_memory": (
        "memory mechanisms regenerating tissue planarian decapitation"
    ),
    "bioelectric_computing": (
        "bioelectric computing biological computation cellular signaling networks"
    ),
}

# Default queries for structural/genomic bio-databases (gene/protein-name
# style queries rather than natural-language phrases). Used for
# --source uniprot|pdb|ncbi_gene when no explicit --topic is given.
GENE_PROTEIN_TOPICS = {
    "innexin": "innexin",
    "connexin": "connexin gap junction",
    "voltage_gated_potassium": "KCNQ voltage-gated potassium channel",
    "voltage_gated_sodium": "SCN voltage-gated sodium channel",
    "voltage_gated_calcium": "CACNA voltage-gated calcium channel",
    "gap_junction_protein": "gap junction protein",
}

# AlphaFold DB is looked up by UniProt accession, not free text — there
# is no sensible default query. Run `acheron collect --source uniprot`
# first to discover accessions for your gene family of interest, then
# pass them with --topic "P17302,Q9Y6N1,..." for --source alphafold.

BIO_DB_SOURCES = ("uniprot", "pdb", "alphafold", "ncbi_gene")

# Default gene names for --source subtiwiki when no --topic is given.
# SubtiWiki (B. subtilis) has its own gene-name space — the GENE_PROTEIN_TOPICS
# terms above are gap-junction/ion-channel oriented and mean nothing to it, so
# it gets its own default list rather than joining BIO_DB_SOURCES. These are
# well-known, canonical B. subtilis regulatory genes, used only as search
# terms — the collector never asserts anything about them beyond what
# SubtiWiki's own API returns.
#
# yugO, mstX, and kinC added because they are the named genes behind the
# biofilm potassium-wave electrical signaling Acheron's own simulations
# (gardner_toggle_circuit.py, native_oscillation_interference.py) already
# cite by mechanism but, until Lundberg, Becker & Choe 2013 (PLoS ONE,
# DOI 10.1371/journal.pone.0060993) was found via this collector, not by
# gene name — see docs cross-reference in the Acheron repo's
# SCIENTIFIC_AUDIT_2026.md, Part 14.
SUBTIWIKI_DEFAULT_GENES = (
    "dnaA", "sigB", "comK", "spo0A", "sinR", "degU", "abrB", "codY",
    "yugO", "mstX", "kinC",
)

# Default search terms for --source planmine when no --topic is given.
# PlanMine's search matches gene ID/symbol OR GO term name (see
# collectors/planmine.py's _search_gene_ids) — mixing well-known
# S. mediterranea gene names with topic words surfaces both specific
# genes and, via GO term matching, ion channel / gap junction annotations
# without needing to already know which gene has them.
PLANMINE_DEFAULT_TOPICS = ("smedwi-1", "djnos", "innexin", "ion channel", "gap junction")


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(verbose: bool) -> None:
    """Nexus — Bioelectric research intelligence (Acheron project)."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# ======================================================================
# COLLECT — harvest papers from sources (Library layer)
# ======================================================================
@main.command()
@click.option(
    "--source",
    type=click.Choice(
        ["pubmed", "biorxiv", "arxiv", "physionet", "uniprot", "pdb", "alphafold", "ncbi_gene", "subtiwiki", "planmine", "all"]
    ),
    default="all",
    help="Which source to collect from ('all' = literature sources only; "
    "bio-databases are collected explicitly since they take gene/protein queries)",
)
@click.option("--topic", "-t", multiple=True, help="Custom search queries")
@click.option("--max-results", "-n", default=50, help="Max papers per query per source")
@click.option("--mindate", default=None, help="Minimum date filter (YYYY or YYYY/MM)")
@click.option("--maxdate", default=None, help="Maximum date filter (YYYY or YYYY/MM)")
@click.option("--download-pdfs", is_flag=True, help="Also download PDFs")
def collect(
    source: str,
    topic: tuple,
    max_results: int,
    mindate: str | None,
    maxdate: str | None,
    download_pdfs: bool,
) -> None:
    """Collect papers into the Library from academic sources and free bio-databases.

    Examples:
        acheron collect --source pubmed -t "bioelectricity planarian" -n 25 --mindate 2015
        acheron collect --source pubmed -t "ion channel voltage" --mindate 2020 --maxdate 2024
        acheron collect --source uniprot -t "innexin" -t "connexin gap junction"
        acheron collect --source pdb -t "connexin gap junction channel"
        acheron collect --source ncbi_gene -t "KCNQ1" -t "GJA1"
        acheron collect --source alphafold -t "P17302,Q9Y6N1"   # UniProt accessions
        acheron collect --source subtiwiki -t "sigB" -t "comK"
        acheron collect --source planmine -t "innexin" -t "smedwi-1"
    """
    from acheron.collectors.arxiv import ArxivCollector
    from acheron.collectors.biorxiv import BiorxivCollector
    from acheron.collectors.physionet import PhysioNetCollector
    from acheron.collectors.planmine import PlanMineCollector
    from acheron.collectors.pubmed import PubMedCollector
    from acheron.collectors.structures import AlphaFoldCollector, PDBCollector
    from acheron.collectors.ncbi_gene import NCBIGeneCollector
    from acheron.collectors.subtiwiki import SubtiWikiCollector
    from acheron.collectors.uniprot import UniProtCollector

    if topic:
        topics = list(topic)
    elif source == "subtiwiki":
        topics = list(SUBTIWIKI_DEFAULT_GENES)
    elif source == "planmine":
        topics = list(PLANMINE_DEFAULT_TOPICS)
    elif source in BIO_DB_SOURCES:
        if source == "alphafold":
            console.print(
                "[yellow]--source alphafold has no default query — pass UniProt "
                "accessions with -t, e.g. -t \"P17302,Q9Y6N1\". "
                "Run --source uniprot first to discover accessions.[/]"
            )
            return
        topics = list(GENE_PROTEIN_TOPICS.values())
    else:
        topics = list(DEFAULT_TOPICS.values())

    collectors = []
    if source in ("pubmed", "all"):
        collectors.append(("PubMed", PubMedCollector()))
    if source in ("biorxiv", "all"):
        collectors.append(("bioRxiv", BiorxivCollector()))
    if source in ("arxiv", "all"):
        collectors.append(("arXiv", ArxivCollector()))
    if source in ("physionet", "all"):
        collectors.append(("PhysioNet", PhysioNetCollector()))
    if source == "uniprot":
        collectors.append(("UniProt", UniProtCollector()))
    if source == "pdb":
        collectors.append(("PDB", PDBCollector()))
    if source == "alphafold":
        collectors.append(("AlphaFold DB", AlphaFoldCollector()))
    if source == "ncbi_gene":
        collectors.append(("NCBI Gene", NCBIGeneCollector()))
    if source == "subtiwiki":
        collectors.append(("SubtiWiki", SubtiWikiCollector()))
    if source == "planmine":
        collectors.append(("PlanMine", PlanMineCollector()))

    total_papers = 0
    total_fulltext = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for coll_name, collector in collectors:
            for query in topics:
                task = progress.add_task(
                    f"[cyan]{coll_name}[/]: {query[:50]}...", total=None
                )
                try:
                    # PubMed collector supports date filtering
                    if isinstance(collector, PubMedCollector):
                        papers = collector.search(
                            query,
                            max_results=max_results,
                            mindate=mindate,
                            maxdate=maxdate,
                        )
                    else:
                        papers = collector.search(query, max_results=max_results)

                    for paper in papers:
                        collector.save_metadata(paper)
                        if download_pdfs and paper.url:
                            collector.download_pdf(paper)
                        if paper.full_text:
                            total_fulltext += 1

                    total_papers += len(papers)
                    desc = f"[green]{coll_name}[/]: {len(papers)} papers"
                    progress.update(task, description=desc)
                except Exception as e:
                    progress.update(task, description=f"[red]{coll_name}[/]: error — {e}")
                    logging.getLogger(__name__).debug("Collection error: %s", e, exc_info=True)
                finally:
                    progress.update(task, completed=True)
            collector.close()

    console.print(
        f"\n[bold green]Library updated: {total_papers} papers collected "
        f"({total_fulltext} with full text).[/]"
    )


# ======================================================================
# INDEX — build vector store from collected papers (Index layer)
# ======================================================================
@main.command()
@click.option("--reindex", is_flag=True, help="Force re-index all papers")
def index(reindex: bool) -> None:
    """Build the Index layer from Library material."""
    from acheron.extraction.chunker import TextChunker
    from acheron.extraction.pdf_parser import PDFParser
    from acheron.models import Paper
    from acheron.vectorstore.store import VectorStore

    settings = get_settings()
    store = VectorStore()
    parser = PDFParser()
    chunker = TextChunker()

    metadata_dir = settings.metadata_dir
    if not metadata_dir.exists():
        console.print("[red]Library empty. Run 'acheron collect' first.[/]")
        return

    metadata_files = list(metadata_dir.glob("*.json"))
    if not metadata_files:
        console.print("[red]No JSON records in Library. Run 'acheron collect' first.[/]")
        return

    console.print(f"Library contains {len(metadata_files)} paper records")

    nxml_dir = settings.data_dir / "nxml"
    total_chunks = 0
    papers_with_text = 0
    papers_abstract_only = 0
    papers_skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Building Index...", total=len(metadata_files))

        for meta_file in metadata_files:
            try:
                paper_data = json.loads(meta_file.read_text(encoding="utf-8"))
                paper = Paper(**paper_data)

                # If we have a PDF, parse it for full text
                if paper.pdf_path and Path(paper.pdf_path).exists():
                    parsed = parser.parse(paper.pdf_path)
                    if parsed:
                        paper.full_text = parsed.full_text
                        paper.sections = parsed.sections
                        paper.tables = parsed.tables

                # If still no sections, try loading NXML from nxml_dir
                if not paper.sections and not paper.full_text and paper.pmcid and nxml_dir.exists():
                    pmcid = paper.pmcid.upper()
                    if not pmcid.startswith("PMC"):
                        pmcid = f"PMC{pmcid}"
                    nxml_path = nxml_dir / f"pmc_{pmcid}.nxml"
                    if nxml_path.exists():
                        paper = _load_nxml_into_paper(paper, nxml_path)

                # Track what we have
                if paper.sections or paper.full_text:
                    papers_with_text += 1
                elif paper.abstract:
                    papers_abstract_only += 1
                else:
                    papers_skipped += 1

                # Chunk and index
                chunks = chunker.chunk_paper(paper)
                if chunks:
                    added = store.add_chunks(chunks)
                    total_chunks += added

            except Exception as e:
                logging.getLogger(__name__).debug("Error indexing %s: %s", meta_file.name, e)

            progress.advance(task)

    console.print(
        f"\n[bold green]Index built: {total_chunks} new chunks. "
        f"Total in store: {store.count()}[/]"
    )
    console.print(
        f"  Full text: {papers_with_text} | "
        f"Abstract only: {papers_abstract_only} | "
        f"Skipped (no content): {papers_skipped}"
    )
    if total_chunks == 0 and store.count() == 0:
        console.print(
            "\n[yellow]0 chunks indexed. Possible causes:[/]\n"
            "  - Papers have no abstract or full text\n"
            "  - No PDFs downloaded and no PMC NXML available\n"
            "  - JSON metadata files are empty or malformed\n"
            "  Run 'acheron collect --source pubmed -t \"your topic\" -n 10' first"
        )


def _load_nxml_into_paper(paper, nxml_path: Path):
    """Load NXML sections into a Paper that lacks full text."""
    import xml.etree.ElementTree as ET

    from acheron.models import PaperSection

    try:
        nxml_text = nxml_path.read_text(encoding="utf-8")
        root = ET.fromstring(nxml_text)
    except (ET.ParseError, OSError) as e:
        logging.getLogger(__name__).debug("Failed to parse NXML %s: %s", nxml_path, e)
        return paper

    sections = []

    # Try to extract body sections
    for body in root.findall(".//body"):
        for sec in body.findall(".//sec"):
            title_el = sec.find("title")
            heading = ""
            if title_el is not None:
                heading = "".join(title_el.itertext()).strip()

            paragraphs = []
            for p in sec.findall(".//p"):
                text = "".join(p.itertext()).strip()
                if text:
                    paragraphs.append(text)

            if paragraphs:
                sections.append(PaperSection(
                    heading=heading,
                    text="\n\n".join(paragraphs),
                ))

    if sections:
        paper.sections = sections
        paper.full_text = "\n\n".join(s.text for s in sections if s.text)

    return paper


# ======================================================================
# QUERY — ask questions (Layer 2+3)
# ======================================================================
@main.command()
@click.argument("question", required=False)
@click.option("--source-filter", "-s", default=None, help="Filter by source")
@click.option("--n-results", "-n", default=10, help="Number of passages to retrieve")
@click.option("--retrieve-only", "-r", is_flag=True, help="Layer 2 only — no Compute")
@click.option("--live", "-l", is_flag=True, help="Enable live retrieval from PubMed/bioRxiv/arXiv")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
def query(
    question: str | None,
    source_filter: str | None,
    n_results: int,
    retrieve_only: bool,
    live: bool,
    interactive: bool,
) -> None:
    """Query the research corpus (Index + Compute layers)."""
    from acheron.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()

    if interactive or question is None:
        _interactive_loop(pipeline, source_filter, n_results, retrieve_only)
        return

    if live and not retrieve_only:
        _run_analyze(pipeline, question, source_filter, n_results, mode=None, live=True)
    else:
        _run_query(pipeline, question, source_filter, n_results, retrieve_only)


def _interactive_loop(pipeline, source_filter, n_results, retrieve_only):
    """Interactive query loop."""
    console.print(
        Panel(
            "[bold cyan]Nexus[/] — System-2 Bioelectric Research Engine\n"
            "Three-layer architecture: Knowledge | Synthesis | Discovery\n\n"
            "Commands:\n"
            "  [dim]Type a question to query (Synthesis + Discovery)[/]\n"
            "  [dim]Prefix with[/] /discover [dim]to run the full discovery loop[/]\n"
            "  [dim]Prefix with[/] /analyze [dim]for the hypothesis engine[/]\n"
            "  [dim]Prefix with[/] /live [dim]for live retrieval from PubMed/bioRxiv/arXiv[/]\n"
            "  [dim]Type[/] quit [dim]to exit[/]",
            title="Acheron Nexus",
            border_style="cyan",
        )
    )
    while True:
        try:
            q = console.input("\n[bold cyan]nexus>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        if q.startswith("/discover "):
            _run_discover(pipeline, q[10:].strip(), source_filter, n_results)
        elif q.startswith("/analyze "):
            _run_analyze(pipeline, q[9:].strip(), source_filter, n_results, mode=None, live=False)
        elif q.startswith("/live "):
            _run_analyze(pipeline, q[6:].strip(), source_filter, n_results, mode=None, live=True)
        else:
            _run_query(pipeline, q, source_filter, n_results, retrieve_only)


def _run_query(pipeline, question, source_filter, n_results, retrieve_only):
    """Execute a single query and display results."""
    with console.status("[bold cyan]Retrieving from Index..."):
        if retrieve_only:
            results = pipeline.retrieve_only(
                question, n_results=n_results, filter_source=source_filter
            )
            _display_retrieval_results(results)
        else:
            response = pipeline.query(
                question, filter_source=source_filter, n_results=n_results
            )
            _display_rag_response(response)


def _run_discover(pipeline, question, source_filter, n_results):
    """Execute the discovery loop and display structured results."""
    from acheron.rag.ledger import ExperimentLedger

    with console.status("[bold cyan]Running discovery loop (Compute layer)..."):
        result = pipeline.discover(
            question, filter_source=source_filter, n_results=n_results
        )

    _display_discovery_result(result)

    # Auto-log to ledger
    ledger = ExperimentLedger()
    entry = ledger.record(result)
    console.print(f"\n[dim]Logged to experiment ledger: {entry.entry_id}[/]")


def _display_retrieval_results(results):
    """Display raw retrieval results (Layer 2 only)."""
    if not results:
        console.print("[yellow]No results in Index.[/]")
        return

    table = Table(title="Retrieved Passages (Layer 2)", show_lines=True)
    table.add_column("#", width=3)
    table.add_column("Score", width=6)
    table.add_column("Paper", max_width=50)
    table.add_column("Section", max_width=15)
    table.add_column("Text", max_width=80)

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            f"{r.relevance_score:.3f}",
            r.paper_title[:50],
            r.section[:15] if r.section else "",
            r.text[:200] + "..." if len(r.text) > 200 else r.text,
        )
    console.print(table)


def _display_rag_response(response):
    """Display a full RAG response with epistemic structure."""
    console.print()

    # Show structured sections if parsed
    has_structured = (
        response.evidence_statements
        or response.inference_statements
        or response.speculation_statements
    )
    if has_structured:
        if response.evidence_statements:
            console.print(Panel(
                "\n".join(f"  - {s}" for s in response.evidence_statements),
                title="[bold green]EVIDENCE[/]",
                border_style="green",
            ))
        if response.inference_statements:
            console.print(Panel(
                "\n".join(f"  - {s}" for s in response.inference_statements),
                title="[bold yellow]INFERENCE[/]",
                border_style="yellow",
            ))
        if response.speculation_statements:
            console.print(Panel(
                "\n".join(f"  - {s}" for s in response.speculation_statements),
                title="[bold red]SPECULATION[/]",
                border_style="red",
            ))
        if response.bioelectric_schematic:
            console.print(Panel(
                response.bioelectric_schematic,
                title="[bold magenta]BIOELECTRIC SCHEMATIC[/]",
                border_style="magenta",
            ))
    else:
        # Fallback: display raw answer
        console.print(Panel(
            Markdown(response.answer),
            title="[bold cyan]Response[/]",
            border_style="cyan",
        ))

    # Source list
    console.print("\n[bold]Sources (Knowledge):[/]")
    seen = set()
    for i, src in enumerate(response.sources, 1):
        if src.paper_id in seen:
            continue
        seen.add(src.paper_id)
        doi_str = f" DOI: {src.doi}" if src.doi else ""
        author_str = ", ".join(src.authors[:3])
        if len(src.authors) > 3:
            author_str += " et al."
        console.print(
            f"  [{i}] {author_str}. [italic]\"{src.paper_title}\"[/].{doi_str}"
        )
    console.print(
        f"\n[dim]Discovery: {response.model_used} | "
        f"Synthesis: {response.total_chunks_searched} chunks searched[/]"
    )


def _display_discovery_result(result):
    """Display a full discovery loop result with all structured fields."""
    console.print()

    # Evidence
    if result.evidence:
        console.print(Panel(
            "\n".join(f"  - {s}" for s in result.evidence),
            title="[bold green]1. EVIDENCE EXTRACTION[/]",
            border_style="green",
        ))

    # Variables
    if result.variables:
        var_table = Table(title="2. BIOELECTRIC VARIABLES", show_lines=True)
        var_table.add_column("Name", style="cyan")
        var_table.add_column("Value")
        var_table.add_column("Unit", style="dim")
        var_table.add_column("Type", style="dim")
        var_table.add_column("Source", style="dim")
        for v in result.variables:
            var_table.add_row(
                v.name, v.value, v.unit, v.variable_type, v.source_ref
            )
        console.print(var_table)

    # Inference / Patterns
    if result.inference:
        console.print(Panel(
            "\n".join(f"  - {s}" for s in result.inference),
            title="[bold yellow]3. PATTERN COMPARISON (Cross-source)[/]",
            border_style="yellow",
        ))

    # Hypotheses
    if result.hypotheses:
        hyp_table = Table(title="4. HYPOTHESES", show_lines=True)
        hyp_table.add_column("Hypothesis", max_width=60)
        hyp_table.add_column("Prior\nConf.", width=8)
        hyp_table.add_column("Predicted Impact", max_width=40)
        hyp_table.add_column("Refs", width=10)
        hyp_table.add_column("Validation", max_width=40)
        for h in result.hypotheses:
            conf_style = {"high": "green", "medium": "yellow", "low": "red"}.get(
                h.confidence, "dim"
            )
            hyp_table.add_row(
                h.statement[:60],
                f"[{conf_style}]{h.confidence}[/{conf_style}]",
                h.predicted_impact[:40] if h.predicted_impact else "",
                ", ".join(h.supporting_refs[:3]),
                h.validation_strategy[:40] if h.validation_strategy else "",
            )
        console.print(hyp_table)

    # Bioelectric Schematic
    if result.bioelectric_schematic:
        console.print(Panel(
            result.bioelectric_schematic,
            title="[bold magenta]5. BIOELECTRIC SCHEMATIC[/]",
            border_style="magenta",
        ))

    # Validation Path
    if result.validation_path:
        console.print(Panel(
            "\n".join(f"  - {s}" for s in result.validation_path),
            title="[bold blue]6. VALIDATION PATH[/]",
            border_style="blue",
        ))

    # Cross-species Notes
    if result.cross_species_notes:
        console.print(Panel(
            "\n".join(f"  - {s}" for s in result.cross_species_notes),
            title="[bold cyan]7. CROSS-SPECIES NOTES[/]",
            border_style="cyan",
        ))

    # Speculation
    if result.speculation:
        console.print(Panel(
            "\n".join(f"  - {s}" for s in result.speculation),
            title="[bold red]SPECULATION[/]",
            border_style="red",
        ))

    # Uncertainty
    if result.uncertainty_notes:
        console.print(Panel(
            "\n".join(f"  - {s}" for s in result.uncertainty_notes),
            title="[bold dim]8. UNCERTAINTY[/]",
            border_style="dim",
        ))

    # Sources
    console.print("\n[bold]Sources (Knowledge):[/]")
    seen = set()
    for i, src in enumerate(result.sources, 1):
        if src.paper_id in seen:
            continue
        seen.add(src.paper_id)
        doi_str = f" DOI: {src.doi}" if src.doi else ""
        author_str = ", ".join(src.authors[:3])
        if len(src.authors) > 3:
            author_str += " et al."
        console.print(
            f"  [{i}] {author_str}. [italic]\"{src.paper_title}\"[/].{doi_str}"
        )

    console.print(
        f"\n[dim]Discovery: {result.model_used} | "
        f"Synthesis: {result.total_chunks_searched} chunks searched[/]"
    )


# ======================================================================
# DISCOVER — run the full discovery loop (dedicated command)
# ======================================================================
@main.command()
@click.argument("question")
@click.option("--source-filter", "-s", default=None, help="Filter by source")
@click.option("--n-results", "-n", default=12, help="Number of passages to retrieve")
@click.option("--tag", "-t", multiple=True, help="Tags for the ledger entry")
@click.option("--notes", default="", help="Notes to attach to the ledger entry")
def discover(
    question: str,
    source_filter: str | None,
    n_results: int,
    tag: tuple,
    notes: str,
) -> None:
    """Run the discovery loop: retrieve, extract, compare, hypothesize, log."""
    from acheron.rag.ledger import ExperimentLedger
    from acheron.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    ledger = ExperimentLedger()

    with console.status("[bold cyan]Executing discovery loop (Compute layer)..."):
        result = pipeline.discover(
            question, filter_source=source_filter, n_results=n_results
        )

    _display_discovery_result(result)

    # Log to ledger
    entry = ledger.record(result, notes=notes, tags=list(tag))
    console.print(f"\n[bold green]Recorded in experiment ledger:[/] {entry.entry_id}")


# ======================================================================
# ANALYZE — Evidence-Bound Hypothesis Engine
# ======================================================================
@main.command()
@click.argument("question")
@click.option("--source-filter", "-s", default=None, help="Filter by source")
@click.option("--n-results", "-n", default=12, help="Number of passages to retrieve")
@click.option("--live", "-l", is_flag=True, help="Enable live retrieval from PubMed/bioRxiv/arXiv")
@click.option(
    "--mode", "-m",
    type=click.Choice(["evidence", "hypothesis", "synthesis", "auto"]),
    default="auto",
    help="Analysis mode (auto detects from query)",
)
@click.option("--tag", "-t", multiple=True, help="Tags for the ledger entry")
@click.option("--notes", default="", help="Notes to attach to the ledger entry")
def analyze(
    question: str,
    source_filter: str | None,
    n_results: int,
    live: bool,
    mode: str,
    tag: tuple,
    notes: str,
) -> None:
    """Evidence-Bound Hypothesis Engine: evidence graph + IBE hypotheses + falsification.

    Modes:
      evidence   — summarize what is known with citations and agreement scores
      hypothesis — generate ranked hypotheses using IBE + falsification
      synthesis  — propose architectures/designs based on evidence + assumptions
      auto       — detect mode from query text (default)

    Examples:
        acheron analyze "What role does Vmem play in planarian regeneration?"
        acheron analyze --mode hypothesis "Why do planaria regenerate heads?"
        acheron analyze --live "bioelectric control of morphogenesis"
    """
    from acheron.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    explicit_mode = mode if mode != "auto" else None

    _run_analyze(pipeline, question, source_filter, n_results, explicit_mode, live)


def _run_analyze(pipeline, question, source_filter, n_results, mode, live):
    """Execute the Evidence-Bound Hypothesis Engine and display results."""
    from acheron.rag.ledger import ExperimentLedger

    with console.status("[bold cyan]Running Evidence-Bound Hypothesis Engine..."):
        result = pipeline.analyze(
            question,
            mode=mode,
            live=live,
            filter_source=source_filter,
            n_results=n_results,
        )

    _display_analysis_result(result)

    # Log to ledger
    ledger = ExperimentLedger()
    # Convert to a DiscoveryResult-compatible format for ledger
    from acheron.models import DiscoveryResult, Hypothesis

    compat_hypotheses = []
    for h in result.hypotheses:
        compat_hypotheses.append(Hypothesis(
            statement=h.statement,
            supporting_refs=h.supporting_refs,
            confidence=(
                "high" if h.confidence >= 70
                else "medium" if h.confidence >= 40
                else "low"
            ),
            predicted_impact="; ".join(h.predictions[:2]) if h.predictions else "",
            assumptions=h.assumptions,
            validation_strategy=h.minimal_test,
        ))

    discovery_compat = DiscoveryResult(
        query=result.query,
        evidence=[
            f"[{c.status.value}] {c.subject} {c.predicate} {c.object}"
            for c in result.evidence_graph.claims
        ],
        hypotheses=compat_hypotheses,
        sources=result.sources,
        model_used=result.model_used,
        total_chunks_searched=result.total_chunks_searched,
        uncertainty_notes=result.uncertainty_notes,
    )
    entry = ledger.record(discovery_compat)
    console.print(f"\n[bold green]Recorded in experiment ledger:[/] {entry.entry_id}")


def _display_analysis_result(result):
    """Display a HypothesisEngineResult with all structured fields."""
    console.print()

    # Mode indicator
    mode_colors = {"evidence": "green", "hypothesis": "purple", "synthesis": "cyan"}
    mode_color = mode_colors.get(result.mode.value, "white")
    console.print(
        f"[bold {mode_color}]MODE: {result.mode.value.upper()}[/{mode_color}]"
        + (f" | Live sources: {result.live_sources_fetched}" if result.live_sources_fetched else "")
    )

    # Evidence Graph — Claims
    if result.evidence_graph.claims:
        claim_table = Table(
            title="EVIDENCE CLAIMS (Knowledge Graph)", show_lines=True
        )
        claim_table.add_column("ID", width=4)
        claim_table.add_column("Claim", max_width=50)
        claim_table.add_column("Status", width=12)
        claim_table.add_column("Agree", width=6, justify="right")
        claim_table.add_column("Refs", width=15)

        for c in result.evidence_graph.claims:
            status_color = {
                "supported": "green",
                "mixed": "yellow",
                "unclear": "dim",
                "unsupported": "red",
            }.get(c.status.value, "dim")
            claim_text = f"{c.subject} {c.predicate} {c.object}".strip() or c.claim_id
            claim_table.add_row(
                c.claim_id,
                claim_text[:50],
                f"[{status_color}]{c.status.value}[/{status_color}]",
                f"{c.agreement_score:.2f}",
                ", ".join(c.source_refs[:4]),
            )
        console.print(claim_table)

    # Evidence Graph — Edges
    if result.evidence_graph.edges:
        console.print(Panel(
            "\n".join(
                f"  {e.source_claim} [{e.relation}] {e.target_claim}"
                for e in result.evidence_graph.edges
            ),
            title="[bold yellow]CLAIM RELATIONSHIPS[/]",
            border_style="yellow",
        ))

    # Ranked Hypotheses with IBE scores + falsification
    if result.hypotheses:
        for h in result.hypotheses:
            if h.confidence >= 70:
                conf_color = "green"
            elif h.confidence >= 40:
                conf_color = "yellow"
            else:
                conf_color = "red"
            content_lines = [
                f"[bold]{h.statement}[/]",
                "",
                f"IBE Scores: explanatory={h.explanatory_power:.1f}  "
                f"simplicity={h.simplicity:.1f}  "
                f"consistency={h.consistency:.1f}  "
                f"mechanistic={h.mechanistic_plausibility:.1f}  "
                f"→ overall=[bold]{h.overall_score:.2f}[/]",
            ]
            if h.rationale:
                content_lines.append(f"\nRationale: {h.rationale}")
            if h.predictions:
                content_lines.append("\n[green]Predictions (if true):[/]")
                for p in h.predictions:
                    content_lines.append(f"  - {p}")
            if h.falsifiers:
                content_lines.append("\n[red]Falsifiers (would disprove):[/]")
                for f in h.falsifiers:
                    content_lines.append(f"  - {f}")
            if h.minimal_test:
                content_lines.append(f"\n[cyan]Minimal test:[/] {h.minimal_test}")
            if h.assumptions:
                content_lines.append(f"\nAssumptions: {'; '.join(h.assumptions)}")
            if h.known_unknowns:
                content_lines.append(f"Known unknowns: {'; '.join(h.known_unknowns)}")
            if h.failure_modes:
                content_lines.append(f"Failure modes: {'; '.join(h.failure_modes)}")
            if h.confidence_justification:
                content_lines.append(f"Justification: {h.confidence_justification}")

            console.print(Panel(
                "\n".join(content_lines),
                title=(
                    f"[bold purple]{h.hypothesis_id} (rank #{h.rank}) — "
                    f"Confidence: [{conf_color}]{h.confidence}/100[/{conf_color}][/]"
                ),
                border_style="purple",
            ))

    # Next queries
    if result.next_queries:
        console.print(Panel(
            "\n".join(f"  - {q}" for q in result.next_queries),
            title="[bold cyan]NEXT BEST EVIDENCE (search queries)[/]",
            border_style="cyan",
        ))

    # Uncertainty
    if result.uncertainty_notes:
        console.print(Panel(
            "\n".join(f"  - {n}" for n in result.uncertainty_notes),
            title="[bold dim]UNCERTAINTY[/]",
            border_style="dim",
        ))

    # Sources
    console.print("\n[bold]Sources (Knowledge):[/]")
    seen = set()
    for i, src in enumerate(result.sources, 1):
        if src.paper_id in seen:
            continue
        seen.add(src.paper_id)
        doi_str = f" DOI: {src.doi}" if src.doi else ""
        author_str = ", ".join(src.authors[:3])
        if len(src.authors) > 3:
            author_str += " et al."
        console.print(
            f"  [{i}] {author_str}. [italic]\"{src.paper_title}\"[/].{doi_str}"
        )

    # Footer
    conf_color = (
        "green" if result.confidence >= 70
        else "yellow" if result.confidence >= 40
        else "red"
    )
    console.print(
        f"\n[dim]Mode: {result.mode.value} | "
        f"Overall confidence: [{conf_color}]{result.confidence}/100[/{conf_color}] | "
        f"Model: {result.model_used} | "
        f"Chunks: {result.total_chunks_searched}"
        + (f" | Live: {result.live_sources_fetched}" if result.live_sources_fetched else "")
        + "[/]"
    )


# ======================================================================
# LEDGER — view experiment ledger
# ======================================================================
@main.command()
@click.option("--tag", "-t", default=None, help="Filter entries by tag")
@click.option("--search", "-s", default=None, help="Search entries by keyword")
@click.option("--export", "-e", default=None, type=click.Path(), help="Export to JSON file")
@click.option("--entry-id", default=None, help="View a specific entry")
@click.option(
    "--record-outcome", default=None,
    help="Attach a real experimental outcome to --entry-id (a MODE 6 prediction "
    "entry). Requires --result.",
)
@click.option(
    "--result", type=click.Choice(["match", "mismatch"]), default=None,
    help="Whether the real outcome matched the prediction. Required with --record-outcome.",
)
@click.option(
    "--force", is_flag=True,
    help="With --record-outcome: overwrite an already-recorded outcome (should be rare).",
)
def ledger(
    tag: str | None,
    search: str | None,
    export: str | None,
    entry_id: str | None,
    record_outcome: str | None,
    result: str | None,
    force: bool,
) -> None:
    """View and search the experiment ledger."""
    from acheron.rag.ledger import ExperimentLedger

    ledger_store = ExperimentLedger()

    if record_outcome is not None:
        if not entry_id:
            console.print("[red]--record-outcome requires --entry-id.[/]")
            return
        if result is None:
            console.print("[red]--record-outcome requires --result [match|mismatch].[/]")
            return
        try:
            entry = ledger_store.record_actual_outcome(
                entry_id, actual_outcome=record_outcome,
                match=(result == "match"), force=force,
            )
        except ValueError as e:
            console.print(f"[red]{e}[/]")
            return
        console.print(f"[bold green]Actual outcome recorded for {entry.entry_id}:[/] {result}")
        _display_ledger_entry(entry)
        return

    if export:
        count = ledger_store.export_all(Path(export))
        console.print(f"[green]Exported {count} entries to {export}[/]")
        return

    if entry_id:
        entry = ledger_store.get_entry(entry_id)
        if not entry:
            console.print(f"[red]Entry not found: {entry_id}[/]")
            return
        _display_ledger_entry(entry)
        return

    if search:
        entries = ledger_store.search_entries(search)
    else:
        entries = ledger_store.list_entries(tag=tag)

    if not entries:
        console.print("[yellow]No ledger entries found.[/]")
        return

    table = Table(title=f"Experiment Ledger ({len(entries)} entries)", show_lines=True)
    table.add_column("ID", max_width=35)
    table.add_column("Timestamp", width=19)
    table.add_column("Query", max_width=50)
    table.add_column("Hypotheses", width=5, justify="right")
    table.add_column("Tags")

    for e in entries:
        table.add_row(
            e.entry_id,
            e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            e.query[:50],
            str(len(e.hypotheses)),
            ", ".join(e.tags) if e.tags else "",
        )
    console.print(table)


def _display_ledger_entry(entry):
    """Display a single ledger entry in detail."""
    console.print(Panel(
        f"[bold]ID:[/] {entry.entry_id}\n"
        f"[bold]Time:[/] {entry.timestamp}\n"
        f"[bold]Query:[/] {entry.query}\n"
        f"[bold]Tags:[/] {', '.join(entry.tags) if entry.tags else '(none)'}\n"
        f"[bold]Notes:[/] {entry.notes or '(none)'}",
        title="[bold cyan]Ledger Entry[/]",
    ))
    if entry.evidence_summary:
        console.print(Panel(
            entry.evidence_summary,
            title="[green]Evidence Summary[/]",
            border_style="green",
        ))
    if entry.hypotheses:
        for i, h in enumerate(entry.hypotheses, 1):
            console.print(f"  [bold]H{i}.[/] [{h.confidence}] {h.statement}")
            if h.validation_strategy:
                console.print(f"      [dim]Validation: {h.validation_strategy}[/]")
    if entry.variables:
        for v in entry.variables:
            console.print(f"  [cyan]{v.name}[/] = {v.value} ({v.unit}) {v.source_ref}")

    if entry.entry_type == "prediction":
        console.print(Panel(
            f"[bold]Organism:[/] {entry.organism}\n"
            f"[bold]Perturbation:[/] {entry.perturbation_gene}\n"
            f"[bold]Predicted confidence:[/] {entry.predicted_confidence_tier} "
            f"(score={entry.predicted_confidence_score:.2f}, "
            f"{entry.predicted_cited_params}/{entry.predicted_total_params} cited)\n"
            f"[bold]Actual outcome:[/] {entry.actual_outcome or '(not yet recorded)'}\n"
            + (
                f"[bold]Match:[/] {'YES' if entry.match else 'NO'} "
                f"(entered {entry.actual_outcome_entered_at})"
                if entry.actual_outcome is not None else ""
            ),
            title="[bold magenta]MODE 6 Prediction[/]",
            border_style="magenta",
        ))
        if entry.prediction_summary:
            console.print(Panel(
                entry.prediction_summary,
                title="[dim]Prediction Summary (as logged)[/]",
                border_style="dim",
            ))

    console.print(f"\n[dim]Source papers: {len(entry.source_ids)}[/]")


# ======================================================================
# REPORT — predicted-vs-actual accuracy (real forward-tested predictions only)
# ======================================================================
@main.command()
@click.option(
    "--accuracy", is_flag=True,
    help="Report real predicted-vs-actual accuracy: overall %, broken down by "
    "organism and by confidence tier (cited vs UNKNOWN parameter reliance).",
)
def report(accuracy: bool) -> None:
    """Report on the experiment ledger's predicted-vs-actual track record.

    This is the ONLY accuracy number this project should show as its real
    predictive track record: it counts exclusively MODE 6 predictions that
    were logged (acheron simulate --model prediction --log) BEFORE a real
    test ran, and later had a real outcome attached (acheron ledger
    --record-outcome). It is never a backtested or literature-matched
    figure -- if nothing has been resolved yet, this says so explicitly
    instead of reporting a different number in its place.
    """
    from acheron.rag.ledger import ExperimentLedger, compute_accuracy_report

    if not accuracy:
        console.print("[yellow]Nothing to report -- pass --accuracy.[/]")
        return

    entries = ExperimentLedger().prediction_entries()
    report_data = compute_accuracy_report(entries)

    if report_data["resolved_count"] == 0:
        console.print(Panel(
            report_data["note"],
            title="[bold yellow]No Resolved Predictions Yet[/]",
            border_style="yellow",
        ))
        unresolved = [e for e in entries if e.actual_outcome is None]
        if unresolved:
            console.print(
                f"\n[dim]{len(unresolved)} prediction(s) logged but awaiting a real "
                f"outcome -- record with 'acheron ledger --entry-id <id> "
                f"--record-outcome \"...\" --result match|mismatch'.[/]"
            )
        return

    overall = report_data["overall"]
    console.print(Panel(
        f"[bold]Resolved predictions:[/] {report_data['resolved_count']}\n"
        f"[bold]Overall accuracy:[/] {overall['accuracy_pct']}% "
        f"({overall['matches']}/{overall['n']} matched)",
        title="[bold cyan]Real Predicted-vs-Actual Accuracy[/]",
        border_style="cyan",
    ))

    organism_table = Table(title="By Organism")
    organism_table.add_column("Organism")
    organism_table.add_column("N", justify="right")
    organism_table.add_column("Matches", justify="right")
    organism_table.add_column("Accuracy", justify="right")
    for organism, bucket in report_data["by_organism"].items():
        organism_table.add_row(
            organism, str(bucket["n"]), str(bucket["matches"]), f"{bucket['accuracy_pct']}%"
        )
    console.print(organism_table)

    tier_table = Table(
        title="By Confidence Tier (cited vs UNKNOWN parameter reliance)"
    )
    tier_table.add_column("Confidence Tier")
    tier_table.add_column("N", justify="right")
    tier_table.add_column("Matches", justify="right")
    tier_table.add_column("Accuracy", justify="right")
    for tier, bucket in report_data["by_confidence_tier"].items():
        tier_table.add_row(
            tier, str(bucket["n"]), str(bucket["matches"]), f"{bucket['accuracy_pct']}%"
        )
    console.print(tier_table)


# ======================================================================
# ADD — manually add a paper (Library layer)
# ======================================================================
@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--title", "-t", default=None, help="Paper title")
@click.option("--doi", default=None, help="Paper DOI")
@click.option("--authors", "-a", default=None, help="Comma-separated authors")
def add(path: str, title: str | None, doi: str | None, authors: str | None) -> None:
    """Add a paper to the Library from a local PDF file."""
    from acheron.extraction.chunker import TextChunker
    from acheron.extraction.pdf_parser import PDFParser
    from acheron.models import PaperSource
    from acheron.vectorstore.store import VectorStore

    parser = PDFParser()
    chunker = TextChunker()
    store = VectorStore()

    with console.status("[bold cyan]Parsing PDF..."):
        paper = parser.parse(path)

    if not paper:
        console.print("[red]Failed to parse PDF.[/]")
        return

    if title:
        paper.title = title
    if doi:
        paper.doi = doi
        paper.paper_id = doi
    if authors:
        paper.authors = [a.strip() for a in authors.split(",")]
    paper.source = PaperSource.MANUAL

    with console.status("[bold cyan]Indexing into vector store..."):
        chunks = chunker.chunk_paper(paper)
        added = store.add_chunks(chunks)

    console.print(
        f"[bold green]Library updated:[/] '{paper.title}' — {added} chunks indexed "
        f"(total: {store.count()})"
    )


# ======================================================================
# STATS — show collection statistics
# ======================================================================
@main.command()
def stats() -> None:
    """Show status of all three layers."""
    from acheron.rag.ledger import ExperimentLedger
    from acheron.vectorstore.store import VectorStore

    settings = get_settings()
    store = VectorStore()
    ledger_store = ExperimentLedger()

    papers = store.list_papers()
    total_chunks = store.count()
    ledger_count = ledger_store.count()

    # Count metadata files in Library
    metadata_dir = settings.metadata_dir
    lib_count = len(list(metadata_dir.glob("*.json"))) if metadata_dir.exists() else 0

    console.print(Panel(
        f"[bold]Layer 1 — Knowledge:[/]\n"
        f"  Paper records:    {lib_count}\n"
        f"  Data dir:         {settings.data_dir}\n\n"
        f"[bold]Layer 2 — Synthesis:[/]\n"
        f"  Papers indexed:   {len(papers)}\n"
        f"  Chunks indexed:   {total_chunks}\n"
        f"  Vector store:     {settings.vectorstore_dir}\n"
        f"  Embedding model:  {settings.embedding_model}\n\n"
        f"[bold]Layer 3 — Discovery:[/]\n"
        f"  LLM provider:     {settings.llm_provider}\n"
        f"  LLM model:        {settings.resolved_llm_model}\n"
        f"  API key:          {'configured' if settings.resolved_llm_api_key else 'NOT SET'}\n"
        f"  Ledger entries:   {ledger_count}",
        title="[bold cyan]Nexus Status[/]",
        border_style="cyan",
    ))

    if papers:
        by_source: dict[str, int] = {}
        for p in papers:
            s = p.get("source", "unknown")
            by_source[s] = by_source.get(s, 0) + 1

        table = Table(title="Library Sources")
        table.add_column("Source")
        table.add_column("Count", justify="right")
        for src, cnt in sorted(by_source.items()):
            table.add_row(src, str(cnt))
        console.print(table)


# ======================================================================
# SIMULATE — Phase 3 parameter-store-driven simulation backends
# ======================================================================
@main.command()
@click.option(
    "--model",
    type=click.Choice(["grn", "prediction"]),
    default="grn",
    help="Simulation backend: 'grn' (gene-regulatory-network only) or "
    "'prediction' (grn + real BETSE bioelectric leg, MODE 6)",
)
@click.option(
    "--organism", required=True,
    help="Organism to simulate, e.g. 'B. subtilis', 'S. mediterranea'",
)
@click.option("--perturbation", required=True, help="Gene to knock down")
@click.option(
    "--knockdown-fraction", default=0.1, type=float,
    help="Fraction of baseline expression remaining after knockdown (0=full knockout, default 0.1)",
)
@click.option(
    "--duration", default=100.0, type=float,
    help="Simulated time duration (dimensionless time units)",
)
@click.option("--n-points", default=101, type=int, help="Number of timepoints to sample")
@click.option(
    "--max-hops", default=2, type=int,
    help="Max graph hops from the perturbed gene to include",
)
@click.option(
    "--channel-class", default=None,
    type=click.Choice(["Na", "K"]),
    help="--model prediction only: ion-channel class to perturb in BETSE. "
    "If omitted, inferred from cited evidence text (e.g. 'potassium'), or "
    "the bioelectric leg runs as an unperturbed baseline.",
)
@click.option(
    "--max-dm-fraction", default=0.1, type=float,
    help="--model prediction only: fraction of default channel conductance "
    "remaining after the perturbation (0=fully blocked, default 0.1)",
)
@click.option(
    "--no-bioelectric", is_flag=True,
    help="--model prediction only: skip the BETSE leg (GRN-only prediction)",
)
@click.option(
    "--output", "-o", type=click.Path(), default=None,
    help="Write full result JSON to this path",
)
@click.option(
    "--log", is_flag=True,
    help="--model prediction only: log this prediction to the experiment ledger "
    "BEFORE any real test runs, so 'acheron ledger --record-outcome' and "
    "'acheron report --accuracy' can later compare it to a real result. "
    "Off by default -- only log the prediction you actually intend to test.",
)
def simulate(
    model: str,
    organism: str,
    perturbation: str,
    knockdown_fraction: float,
    duration: float,
    n_points: int,
    max_hops: int,
    channel_class: str | None,
    max_dm_fraction: float,
    no_bioelectric: bool,
    output: str | None,
    log: bool,
) -> None:
    """Simulate a gene perturbation's downstream effect using Phase 3's cited parameter store.

    Confidence scales with what fraction of the simulated network's edges carry
    a cited rate constant (from extracted ParameterRecords) vs. an explicitly
    flagged placeholder default -- never a silently invented number.

    Examples:
        acheron simulate --model grn --organism "B. subtilis" --perturbation yugO
        acheron simulate --model prediction --organism "B. subtilis" --perturbation yugO
    """
    from acheron.simulation.betse_model import BETSEBackendUnavailable
    from acheron.simulation.grn_model import GRNBackendUnavailable

    if model == "grn":
        from acheron.simulation.grn_model import simulate_grn

        try:
            status_msg = f"[bold cyan]Simulating GRN perturbation of '{perturbation}'..."
            with console.status(status_msg):
                result = simulate_grn(
                    organism=organism,
                    perturbation_gene=perturbation,
                    knockdown_fraction=knockdown_fraction,
                    duration=duration,
                    n_points=n_points,
                    max_hops=max_hops,
                )
        except GRNBackendUnavailable as e:
            console.print(f"[red]{e}[/]")
            return
        except ValueError as e:
            console.print(f"[red]{e}[/]")
            return

        _display_grn_result(result)

        if output:
            Path(output).write_text(result.model_dump_json(indent=2), encoding="utf-8")
            console.print(f"\n[dim]Full result (including trajectories) written to {output}[/]")
        return

    # model == "prediction"
    from acheron.rag.hypothesis_engine import format_prediction_context, run_prediction_mode

    try:
        status_msg = (
            f"[bold cyan]Running combined GRN+BETSE prediction for '{perturbation}' "
            f"(this runs real BETSE simulations, can take ~1-2 minutes)..."
        )
        with console.status(status_msg):
            result = run_prediction_mode(
                organism=organism,
                perturbation_gene=perturbation,
                channel_class=channel_class,
                max_dm_fraction=max_dm_fraction,
                knockdown_fraction=knockdown_fraction,
                run_bioelectric=not no_bioelectric,
                duration=duration,
                n_points=n_points,
                max_hops=max_hops,
            )
    except (GRNBackendUnavailable, BETSEBackendUnavailable) as e:
        console.print(f"[red]{e}[/]")
        return
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        return

    _display_prediction_result(result)

    if output:
        Path(output).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"\n[dim]Full result (including trajectories) written to {output}[/]")

    if log:
        from acheron.rag.ledger import ExperimentLedger

        bio = result.bioelectric
        cited_params = result.grn.n_cited_edges + (
            1 if bio and bio.channel_class_cited else 0
        )
        total_params = result.grn.n_total_edges + (1 if bio else 0)
        entry = ExperimentLedger().record_prediction(
            organism=result.organism,
            perturbation_gene=result.perturbation_gene,
            confidence_score=result.overall_confidence_score,
            confidence_tier=result.overall_confidence_tier.value,
            cited_params=cited_params,
            total_params=total_params,
            prediction_summary=format_prediction_context(result),
        )
        console.print(
            f"\n[bold green]Logged to experiment ledger:[/] {entry.entry_id}\n"
            f"[dim]Record the real outcome later with: acheron ledger "
            f"--entry-id {entry.entry_id} --record-outcome \"...\" --result match|mismatch[/]"
        )


def _display_prediction_result(result) -> None:
    """Display a PredictionModeResult (MODE 6): combined GRN+BETSE outcome,
    the cited-vs-assumed split, and the fed-through experiment protocol."""
    tier_colors = {"high": "green", "medium": "yellow", "low": "red"}
    tier_color = tier_colors.get(result.overall_confidence_tier.value, "dim")
    bio = result.bioelectric
    bio_line = (
        f"Bioelectric: Vmem {bio.baseline_vmem_mV:.2f} -> {bio.perturbed_vmem_mV:.2f} mV "
        f"(channel {bio.channel_class or 'none'}, "
        f"{'cited' if bio.channel_class_cited else 'assumed'})"
        if bio else "Bioelectric: did not run (see warnings)"
    )
    console.print(Panel(
        f"[bold]Organism:[/] {result.organism}\n"
        f"[bold]Perturbation:[/] {result.perturbation_gene}\n"
        f"[bold]GRN:[/] {result.grn.n_cited_edges}/{result.grn.n_total_edges} edges cited\n"
        f"[bold]{bio_line}[/]\n"
        f"[bold]Overall confidence:[/] "
        f"[{tier_color}]{result.overall_confidence_tier.value}[/{tier_color}] "
        f"(score={result.overall_confidence_score:.2f})",
        title="[bold cyan]Prediction (MODE 6): GRN + Bioelectric[/]",
        border_style="cyan",
    ))

    if result.warnings:
        console.print(Panel(
            "\n".join(f"  - {w}" for w in result.warnings),
            title="[bold yellow]WARNINGS[/]",
            border_style="yellow",
        ))

    console.print(Panel(
        "\n".join(f"  - {c}" for c in result.cited_predictions) or "  (none)",
        title="[bold green]PREDICTED FROM CITED DATA[/]",
        border_style="green",
    ))
    console.print(Panel(
        "\n".join(f"  - {a}" for a in result.assumed_predictions) or "  (none)",
        title="[bold red]PREDICTED FROM ASSUMED DEFAULTS[/]",
        border_style="red",
    ))

    if result.experiment_protocol:
        protocol_text = result.experiment_protocol
        if result.experiment_protocol_caveat:
            protocol_text += f"\n\nCAVEAT: {result.experiment_protocol_caveat}"
        console.print(Panel(
            protocol_text,
            title="[bold blue]WET-LAB PROTOCOL (experiment_designer.py)[/]",
            border_style="blue",
        ))
    else:
        console.print(
            f"\n[dim]No wet-lab protocol proposed: {result.experiment_protocol_caveat}[/]"
        )


def _display_grn_result(result) -> None:
    """Display a GRNSimulationResult: network edges, confidence, trajectory summary."""
    tier_colors = {"high": "green", "medium": "yellow", "low": "red"}
    tier_color = tier_colors.get(result.confidence_tier.value, "dim")
    console.print(Panel(
        f"[bold]Organism:[/] {result.organism}\n"
        f"[bold]Perturbation:[/] {result.perturbation_gene} "
        f"(held at {result.knockdown_fraction:.0%} of baseline)\n"
        f"[bold]Network:[/] {len(result.genes)} gene(s), {result.n_total_edges} edge(s)\n"
        f"[bold]Confidence:[/] [{tier_color}]{result.confidence_tier.value}[/{tier_color}] "
        f"({result.n_cited_edges}/{result.n_total_edges} edges cited, "
        f"score={result.confidence_score:.2f})",
        title="[bold cyan]GRN Simulation[/]",
        border_style="cyan",
    ))

    if result.warnings:
        console.print(Panel(
            "\n".join(f"  - {w}" for w in result.warnings),
            title="[bold yellow]WARNINGS[/]",
            border_style="yellow",
        ))

    if result.edges:
        table = Table(title="Network Edges", show_lines=True)
        table.add_column("Source", style="cyan")
        table.add_column("Relationship")
        table.add_column("Target", style="cyan")
        table.add_column("Sign")
        table.add_column("Rate", justify="right")
        table.add_column("Cited?")
        for e in result.edges:
            rate_style = "green" if e.rate_cited else "red"
            table.add_row(
                e.source_gene, e.relationship_type, e.target_gene,
                e.sign, f"{e.rate_constant:g}",
                f"[{rate_style}]{'yes' if e.rate_cited else 'NO (assumed)'}[/{rate_style}]",
            )
        console.print(table)

    if result.trajectories:
        traj_table = Table(title="Trajectory Summary (first -> last timepoint)")
        traj_table.add_column("Gene", style="cyan")
        traj_table.add_column("Start", justify="right")
        traj_table.add_column("End", justify="right")
        traj_table.add_column("Delta", justify="right")
        for gene, values in sorted(result.trajectories.items()):
            start, end = values[0], values[-1]
            traj_table.add_row(gene, f"{start:.3f}", f"{end:.3f}", f"{end - start:+.3f}")
        console.print(traj_table)
        console.print(
            f"[dim]{len(result.timepoints)} timepoints simulated over duration; "
            f"use --output for full time series.[/]"
        )

    if result.unknown_parameters:
        console.print(Panel(
            "\n".join(f"  - {u}" for u in result.unknown_parameters),
            title="[bold red]UNKNOWN / ASSUMED PARAMETERS (not silently defaulted)[/]",
            border_style="red",
        ))

    if result.notes:
        console.print(Panel(
            "\n".join(f"  - {n}" for n in result.notes),
            title="[bold dim]MODELING NOTES[/]",
            border_style="dim",
        ))


# ======================================================================
# SERVE — web UI
# ======================================================================
@main.command()
@click.option("--host", default=None, help="Override host")
@click.option("--port", "-p", default=None, type=int, help="Override port")
def serve(host: str | None, port: int | None) -> None:
    """Start the web interface."""
    import uvicorn

    from acheron.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "acheron.web.app:app",
        host=host or settings.host,
        port=port or settings.port,
        reload=False,
    )


# ======================================================================
# INTERFACE — Nexus voice/avatar kiosk
# ======================================================================
@main.command()
@click.option("--host", default=None, help="Override interface host")
@click.option("--port", "-p", default=None, type=int, help="Override interface port")
@click.option("--whisper-model", default=None, help="Whisper model size (tiny/base/small/medium/large-v3)")
@click.option("--whisper-device", default=None, help="Whisper device (auto/cpu/cuda)")
@click.option("--piper-model", default=None, help="Path to Piper TTS ONNX model")
def interface(
    host: str | None,
    port: int | None,
    whisper_model: str | None,
    whisper_device: str | None,
    piper_model: str | None,
) -> None:
    """Start the Nexus voice/avatar interface (Arthur layer).

    Launches a WebSocket server with the kiosk frontend for voice-driven
    interaction with Acheron's RAG pipeline.  Supports Whisper STT and
    Piper/ElevenLabs TTS.

    Open the displayed URL in a browser (or Chromium kiosk mode) to
    interact with the fullscreen interface.
    """
    import uvicorn

    from acheron.config import get_settings
    from acheron.interface.ws_server import create_interface_app

    settings = get_settings()
    app = create_interface_app(
        stt_model=whisper_model or settings.whisper_model,
        stt_device=whisper_device or settings.whisper_device,
        piper_model=piper_model or settings.piper_model_path,
        elevenlabs_key=settings.elevenlabs_api_key,
        elevenlabs_voice=settings.elevenlabs_voice_id,
        session_path=settings.nexus_session_path,
    )

    bind_host = host or settings.interface_host
    bind_port = port or settings.interface_port
    console.print(Panel(
        f"[bold cyan]NEXUS Interface[/bold cyan]\n"
        f"Kiosk:  http://{bind_host}:{bind_port}/\n"
        f"WebSocket:  ws://{bind_host}:{bind_port}/ws\n"
        f"Whisper: {whisper_model or settings.whisper_model}\n"
        f"TTS: {'Piper' if (piper_model or settings.piper_model_path) else 'ElevenLabs' if settings.elevenlabs_api_key else 'disabled'}",
        title="Starting Interface",
        border_style="cyan",
    ))

    uvicorn.run(app, host=bind_host, port=bind_port, reload=False)


if __name__ == "__main__":
    main()
