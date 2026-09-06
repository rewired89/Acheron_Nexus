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
full CLI reference, `MANIFEST.md` / `LOGIC_HASH.md` for the project's
non-negotiable rules (read these before touching `rag/` or any collector),
and `TROUBLESHOOTING.md` for known Compute-layer / local-environment
failure modes (recurring "Compute layer unavailable" errors are almost
never the same root cause twice).

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
  `science_filter._ORGANISM_TIERS` has a `"bacteria"` tier (added alongside
  this module) so SubtiWiki/B. subtilis records score `organism_match`
  against their own organism instead of being silently penalized for not
  matching a planarian-only scorer — see `_ORGANISM_TARGET_MAP` in
  `parameter_extractor.py` for the organism-string-to-tier mapping; extend
  that map (not `score_organism_match` itself) when a new curated-database
  organism is added. Run `python scripts/validate_parameters.py` after
  indexing to see what fraction of extracted records carry a cited
  rate/affinity value versus an explicit `UNKNOWN`, broken down per organism.
- **GRN simulation (`simulation/grn_model.py`, `acheron simulate --model grn`)**
  consumes `ParameterRecord`s (Phase 3's output) to simulate a gene
  knockdown's downstream effect via Antimony/libRoadRunner (the ODE engine
  Tellurium wraps — installed directly via the `simulation` extra,
  `pip install -e ".[simulation]"`, to skip Tellurium's own plotting/Jupyter
  dependencies). It reuses `parameter_extractor.py`'s `ParameterStore` and
  `ConfidenceTier` rather than building a second storage layer or a second
  confidence vocabulary, same reuse discipline as `parameter_extractor.py`
  itself follows relative to `science_filter.py`. Every simulated edge's
  rate constant and activation/repression sign is either read verbatim from
  a cited source (a real `rate_or_affinity` value, or an explicit
  positive/negative keyword in the record's own `evidence_text`) or an
  engineering placeholder flagged in `unknown_parameters` — never silently
  defaulted — and `confidence_score` is exactly the fraction of the
  simulated network's edges carrying a cited rate constant. This module
  only reads from the parameter store; it does not modify `extraction/`,
  `rag/`, or the top-level `simulations/` directory (Acheron-side BETSE
  scripts, unaffected by this Nexus-side addition).
- **Bioelectric simulation (`simulation/betse_model.py`)** wraps a real
  `betse` CLI run (not a reimplemented approximation) to predict a Vmem
  shift from an ion-channel-class perturbation, run twice (baseline +
  perturbed) so `delta_vmem_mV` is a real difference between two full
  BETSE runs, not a single-run guess. Only Na/K are supported (the only
  channel classes with a pre-populated dynamic entry in BETSE's own
  default config template) — Ca/Cl raise `ChannelClassUnsupported` rather
  than hand-authoring biophysical parameters with no cited source.
  `infer_channel_class()` maps a gene to a channel class only via a
  literal keyword match against that gene's own cited `evidence_text`
  (e.g. "potassium", "k+") — never from the gene name alone, since BETSE
  itself has no concept of genes, only ion-channel biophysics.
- **MODE 6 / Prediction (`rag/hypothesis_engine.py`'s `run_prediction_mode()`,
  `acheron simulate --model prediction`)** is the one hypothesis-engine
  mode that is not purely an LLM prompt: it deterministically runs
  `simulation/grn_model.py` and `simulation/betse_model.py` for the same
  perturbation, then derives ONE confidence score as (cited GRN edges +
  1-if-the-bioelectric-channel-mapping-was-cited) / (total GRN edges +
  1-if-the-bioelectric-leg-ran) — never a separately-judged number, and
  never counting `decay_rate`/`knockdown_fraction`/`max_dm_fraction`
  themselves (those are engineering dials, not cited-or-not evidence).
  `PREDICTION_PROMPT` (MODE 6's system prompt) instructs the LLM to
  narrate this already-computed result, explicitly split into "Predicted
  From Cited Data" vs. "Predicted From Assumed Defaults" sections, and
  forbids it from recomputing or inventing any number — the LLM call is
  optional narration around real numbers, not their source.
  `run_prediction_mode()` feeds `required_measurements=["vmem"]` (when the
  bioelectric leg ran) into the existing `experiment_designer.py`'s
  `propose_experiment()`; since that module's template catalog is
  planarian-only (`vmem_imaging_post_amputation()` /
  `gap_junction_modulation_regeneration()`), a returned protocol for a
  non-planarian organism (e.g. B. subtilis) carries an explicit
  organism-mismatch caveat rather than being presented as directly
  applicable. `NexusMode.PREDICTION` lives in `models.py` alongside the
  other five modes; `PredictionModeResult`/`_tier_for_fraction()` are kept
  local to `hypothesis_engine.py`, same convention `parameter_extractor.py`
  and `grn_model.py` use for their own result shapes.
- **Predicted-vs-actual tracking (`rag/ledger.py`, `acheron report
  --accuracy`)** is the ONLY accuracy figure this project may present as
  its real predictive track record — never a backtested or literature-
  matched number reported as if it were this one. A MODE 6 prediction is
  logged via `ExperimentLedger.record_prediction()` (plain values only:
  organism, gene, confidence score/tier, cited-vs-total parameter counts,
  and `format_prediction_context(result)` as the summary — never the
  `PredictionModeResult` object itself, so `ledger.py` stays a light leaf
  module `hypothesis_engine.py`/`cli.py` depend on, not the reverse) at
  `acheron simulate --model prediction --log` time, which sets that
  entry's `timestamp` BEFORE any real test runs. A real, manually-entered
  result is attached later via `acheron ledger --entry-id <id>
  --record-outcome "..." --result match|mismatch`, which sets
  `actual_outcome_entered_at` and refuses (unless `--force`) to silently
  overwrite an already-resolved entry, and refuses entirely on any entry
  that isn't `entry_type == "prediction"`. `compute_accuracy_report()`
  computes accuracy ONLY over entries that have been through both calls;
  with zero resolved entries it returns an explicit explanatory note
  instead of a number, and `acheron report --accuracy` prints that note
  rather than inventing a placeholder percentage. Its "cited vs UNKNOWN
  parameters" breakdown reuses `predicted_confidence_tier`
  (high/medium/low), the same cited-ratio bucketing used everywhere else
  in this codebase, rather than a second threshold invented for this
  report alone. `--log` on `simulate --model prediction` defaults to
  off — only the prediction a user actually intends to test in the wet
  lab should ever enter this ledger, not every parameter-sweep run.
