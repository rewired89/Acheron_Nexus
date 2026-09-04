# CLAUDE.md

Guidance for Claude Code sessions working in this repository.

## What this is

Acheron Nexus is a domain-specific RAG research assistant for bioelectricity
and biomedical research (planarian regeneration, ion channels, gap
junctions/innexins, EEG). It collects from literature sources (PubMed,
bioRxiv, arXiv, PhysioNet) and curated bio-databases (UniProt, RCSB PDB,
AlphaFold DB, NCBI Gene, SubtiWiki, PlanMine), indexes everything into a local
ChromaDB vector store, and answers questions with grounded, cited
responses through a FastAPI web UI.

See `CODEMAP.md` for the module-by-module file map, `README.md` for the
full CLI reference, and `MANIFEST.md` / `LOGIC_HASH.md` for the project's
non-negotiable rules (read these before touching `rag/` or any collector).

## The one rule that overrides everything else

**No invented facts, numbers, or citations, ever.** Every number, claim, or
field this codebase surfaces must trace back to something a real source
(a paper, a database API response) actually said. If a value isn't
present, say so (`"not recorded"`, `"UNKNOWN — needs measurement"`) —
never fill a gap with something plausible-sounding. This applies to
collector parsing (only extract fields the API actually returned) exactly
as much as it applies to the RAG pipeline's generated answers. When adding
a new data source, if you can't verify the real API shape (endpoints,
field names) from actual documentation or a live response, stop and say
so rather than guessing — a wrong guess here isn't a style issue, it
either silently corrupts data or crashes at runtime.

## Quickstart

```bash
pip install -e ".[dev]"
cp .env.example .env   # then set ANTHROPIC_API_KEY
acheron collect        # harvest literature (all sources) into the Library
acheron index          # build the vector store from the Library
acheron serve          # web UI at http://127.0.0.1:8000
```

Run the test suite with:
```bash
pytest
```
Tests are offline and fixture-based (see `tests/test_bio_collectors.py`,
`tests/test_subtiwiki_collector.py`, and `tests/test_planmine_collector.py`
for the pattern collectors follow) —
no live network calls, no live LLM calls. A `VectorStore` test does spin up
a real ChromaDB instance with local embeddings, which is slow on first run
(downloads/verifies the embedding model) but makes no external API calls
beyond that.

## Conventions worth knowing before editing

- **Prose style**: no em dashes anywhere Nexus's own text reaches a
  reader — the UI copy and every LLM system prompt (`QUERY_SYSTEM_PROMPT`,
  the Discovery Engine `SYSTEM_PROMPT`, the `hypothesis_engine.py` mode
  prompts) explicitly instruct against them. Use a comma, period, or
  "and"/"but" instead. This is a standing project preference, not a
  one-off request — keep it when writing new prompts or UI text.
- **The Query tab vs. Discover/Analyze**: `pipeline.py`'s `query()` uses a
  deliberately short, plain-English prompt (`QUERY_SYSTEM_PROMPT` +
  `SHORT_QUERY_TEMPLATE`) distinct from the much heavier Discovery Engine
  `SYSTEM_PROMPT` used by `discover()`/`analyze()`. Don't merge these back
  together — the split exists because sharing one heavy prompt made every
  quick question come back as a multi-page research report.
- **Model config**: the Anthropic model string lives in exactly one place,
  `Settings.anthropic_model` in `config.py` (default `claude-sonnet-5`).
  If Anthropic retires a model, that's the only line that needs to change,
  plus `.env.example`. Anthropic API calls in `pipeline.py` intentionally
  omit `temperature` — recent Claude models reject it.
- **Collectors** all follow one interface (`BaseCollector.search()`); see
  the "Adding a new collector" checklist in `CODEMAP.md` before writing a
  new one.
- **`organism` / `source_type` fields** on `Paper`/`TextChunk` distinguish
  curated-database records (`SourceType.CURATED_DB`, e.g. UniProt,
  SubtiWiki, PlanMine) from literature (`SourceType.LITERATURE`). Set
  these on any new non-literature collector's `Paper` objects, and
  they'll flow through `chunker.py` to ChromaDB metadata automatically.
- **Curated-database sources are not equally complete.** SubtiWiki is a
  hand-curated wiki with regulon/pathway structure; PlanMine (InterMine-
  based) is sequence/annotation-centric with no phenotype, pathway, or
  regulon classes at all — see `collectors/planmine.py`'s module
  docstring for the specifics. Any future confidence-scoring logic over
  collected records needs to account for this per-source completeness
  asymmetry rather than treating "field is empty" the same way across
  every curated database.
- **Do not touch `rag/pipeline.py`, `rag/hypothesis_engine.py`, or
  `rag/ledger.py`** when the task is scoped to a new data collector —
  those are Discovery/Compute-layer files with their own review bar; a
  new collector only needs Library-layer (`collectors/`) and
  Index-layer (`extraction/`, `vectorstore/`) changes plus `models.py`
  schema additions and `cli.py` wiring.
- **Parameter extraction (`extraction/parameter_extractor.py`)** turns
  indexed RAG chunks into structured `(subject) -[relationship]->
  (object)` records for downstream simulation use — see the module's own
  docstring for the full design. It deliberately reuses
  `rag/science_filter.py`'s `score_evidence()` as its only confidence
  source (bucketed into HIGH/MEDIUM/LOW) rather than building a second
  scoring system, and stores records as flat JSON files under
  `data_dir/parameters/`, mirroring `rag/ledger.py`'s file-per-entry
  pattern rather than mixing them into ChromaDB's text-chunk collection.
  A known, intentionally-not-patched limitation:
  `science_filter._ORGANISM_TIERS` has no bacterial tier, so every
  SubtiWiki-derived record scores `organism_match=0.0` regardless of its
  real scientific merit, while PlanMine records correctly match the
  existing "planarian" tier — `scripts/validate_parameters.py` reports
  this asymmetry every run rather than hiding it. Run
  `python scripts/validate_parameters.py` after indexing to see what
  fraction of extracted records carry a cited rate/affinity value versus
  an explicit `UNKNOWN`, broken down per organism.
