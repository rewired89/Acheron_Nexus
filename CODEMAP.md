# Acheron Nexus — Code Map

One-line-per-module map of `src/acheron/`. See `README.md` for the CLI and
`MANIFEST.md` / `LOGIC_HASH.md` for the science-first rules every layer
must respect (no invented numeric or factual claims, every value traceable
to a citation or a real API response).

## Top level

| Path | Purpose |
|---|---|
| `src/acheron/cli.py` | `acheron` CLI entry point: `collect`, `index`, `query`, `serve`, `stats`, `add`, `interface` |
| `src/acheron/config.py` | Central `Settings` (env vars / `.env`) — LLM provider, model, data dirs, API keys |
| `src/acheron/models.py` | Core domain models: `Paper`, `TextChunk`, `QueryResult`, `RAGResponse`, enums (`PaperSource`, `SourceType`, `NexusMode`, ...) |

## `collectors/` — Layer 1 (Knowledge): harvest raw records into the Library

| File | Source | Kind |
|---|---|---|
| `base.py` | — | Abstract `BaseCollector`: HTTP client, retry, PDF download, JSON metadata save |
| `pubmed.py` | PubMed / PMC (NCBI E-Utilities) | Literature |
| `biorxiv.py` | bioRxiv / medRxiv | Literature (preprints) |
| `arxiv.py` | arXiv (q-bio) | Literature (preprints) |
| `physionet.py` | PhysioNet | Datasets (EEG/ECG) |
| `uniprot.py` | UniProtKB | Curated protein function/sequence |
| `structures.py` | RCSB PDB + AlphaFold DB | Curated 3D structure data |
| `ncbi_gene.py` | NCBI Gene | Curated genomic context |
| `subtiwiki.py` | SubtiWiki (uni-goettingen.de) | Curated *B. subtilis* gene/protein/regulatory data — locus tags, function, protein-protein interactions, regulon/TF relationships, per-condition expression values |
| `planmine.py` | PlanMine (planmine.mpinat.mpg.de, InterMine-based) | *S. mediterranea* gene/homology/RNAi-expression/protein-domain data — see the module docstring's "DATA COMPLETENESS ASYMMETRY vs. SubtiWiki" section before using this data in any confidence-scoring logic: no phenotype class, no reliable gene symbols, no pathway/regulon structure exist in this source at all |

Every collector implements `search(query, max_results) -> list[Paper]` and
converts its source's native response into a `Paper` whose `abstract` is a
structured plain-text summary built only from fields the source actually
returned (see each file's module docstring for exactly which fields, and
whether the source is keyless).

## `extraction/` — turns a `Paper` into embeddable chunks

| File | Purpose |
|---|---|
| `chunker.py` | `TextChunker.chunk_paper(Paper) -> list[TextChunk]`; carries `organism`/`source_type`/`url`/`source` from `Paper` onto each chunk's metadata |
| `pdf_parser.py` | PyMuPDF/pdfplumber-based PDF text + table extraction |
| `parameter_extractor.py` | `extract_from_chunk(TextChunk) -> list[ParameterRecord]` / `extract_from_store(VectorStore) -> list[ParameterRecord]`: turns indexed chunks into structured `(subject) -[relationship_type]-> (object)` facts with a cited `rate_or_affinity` or explicit `UNKNOWN`. Confidence tier is a thin bucketing of `rag/science_filter.py`'s existing `score_evidence()` — no second scorer. `ParameterStore` persists records as flat JSON under `data_dir/parameters/` (mirrors `rag/ledger.py`'s pattern). Source-aware: SubtiWiki- and PlanMine-specific regex parsers target those two collectors' own known structured-text layout; every other source falls back to a heuristic literature parser (flagged `heuristic=True`). |

## `simulation/` — consumes Phase 3's ParameterRecords to run perturbation simulations

| File | Purpose |
|---|---|
| `grn_model.py` | `simulate_grn(organism, perturbation_gene, ...) -> GRNSimulationResult`: builds a small gene-regulatory network from `extraction/parameter_extractor.py`'s `ParameterRecord`s (SubtiWiki/PlanMine-eligible relationship types only — homology/domain/RNAi-expression annotations and SubtiWiki's non-specific regulon-summary records are excluded), converts it to SBML via Antimony, and integrates it with libRoadRunner (the same engine Tellurium wraps) to show how a gene knockdown propagates over time. Every edge's rate constant is either a literal cited number from `rate_or_affinity` or an explicitly-flagged placeholder — never a silent default — and `confidence_score` is exactly the fraction of simulated edges that carried a cited value. Optional dependency group: `pip install -e ".[simulation]"` (antimony + libroadrunner); raises `GRNBackendUnavailable` with that install line if missing, rather than failing on decoy `nan`s or invented numbers. Does not touch the top-level `simulations/` directory (Acheron-side BETSE scripts) or `rag/pipeline.py`. |
| `betse_model.py` | `simulate_bioelectric(channel_class, ...) -> BioelectricSimulationResult`: shells out to a real `betse` CLI run (config → seed → init → sim, twice — unperturbed baseline and channel-perturbed) and parses BETSE's own logged "Final average cell Vmem", never computing a Vmem itself. Only Na/K channel classes are supported (the only ones with a pre-populated dynamic channel entry in BETSE's own default template — Ca/Cl would need hand-authored biophysical parameters this module has no cited source for, so it raises `ChannelClassUnsupported` instead of guessing). `infer_channel_class()` maps a gene's cited `evidence_text` to a channel class via literal keyword match only (e.g. "potassium"/"k+") — never from the gene name alone. Optional dependency: `pip install betse` (also in the `simulation` extra); raises `BETSEBackendUnavailable` if missing. Feeds `rag/hypothesis_engine.py`'s MODE 6 (prediction) alongside `grn_model.py`. |

`rag/hypothesis_engine.py`'s MODE 6 (`NexusMode.PREDICTION`, `run_prediction_mode()`) combines both modules' outputs into one `PredictionModeResult` with a single confidence score derived only from the cited-vs-total parameter ratio across both legs (see that module's own section below) — it is the consumer of this directory, not a third file in it.

## `vectorstore/` — Layer 2 (Index): embedding storage and retrieval

| File | Purpose |
|---|---|
| `store.py` | `VectorStore`: ChromaDB-backed. `add_chunks()` (dedup by `chunk_id`), `search()` (semantic query -> ranked `QueryResult`), `list_papers()`, `count()` |

## `rag/` — Layer 3 (Discovery/Compute): retrieval, synthesis, hypothesis generation

| File | Purpose |
|---|---|
| `pipeline.py` | `RAGPipeline`: `query()` (short, plain-English answers — the "Query" tab), `discover()` (full Discovery Engine loop), `analyze()` (routes to hypothesis_engine mode prompts), `retrieve_only()` |
| `hypothesis_engine.py` | Evidence-Bound Hypothesis Engine: the six `NexusMode` prompts (evidence/hypothesis/synthesis/decision/tutor/**prediction**) + fast-decision endpoint. MODE 6 (`run_prediction_mode()`) is the one mode that isn't purely LLM-prompt-driven — it deterministically calls `simulation/grn_model.py` + `simulation/betse_model.py`, derives one confidence score from their combined cited-vs-total parameter ratio (never a separate judgment call), feeds the result into `experiment_designer.py`, and only then hands the whole thing to `PREDICTION_PROMPT` for the LLM to narrate (never recompute). |
| `science_filter.py` | Science-First evidence scoring/filtering, organism-strictness ranking |
| `query_parser.py` | Query Understanding (Stage A) for Science-First Mode |
| `live_retrieval.py` | On-demand PubMed/bioRxiv/arXiv fetch when local evidence is weak |
| `experiment_designer.py` | Minimal Viable Experiment (MVE) proposal generation |
| `bio_units.py` | Biological Information Module (BIM) quantitative spec framework |
| `ledger.py` | Persistent log of discovery-loop findings (the "Ledger" tab), plus predicted-vs-actual tracking for MODE 6 predictions: `record_prediction()` logs a prediction's organism/gene/confidence/cited-vs-total counts BEFORE a real test runs; `record_actual_outcome()` attaches a real, manually-entered result afterward (refuses to touch a non-prediction entry or silently overwrite one already resolved); `compute_accuracy_report()` computes real predicted-vs-actual accuracy — by organism and by confidence tier — over ONLY entries that went through both calls (see `acheron report --accuracy`). Takes plain values, not `PredictionModeResult`, so this stays a light leaf module `rag/hypothesis_engine.py` depends on, not the reverse. |

## `reasoning/` — standalone quantitative/simulation modules (numbered per an internal spec)

Independent analytical modules (denram, freeze_thaw, heterogeneity, homeostatic,
mosaic, ribozyme, sensitivity, state_space, substrate, validation, empirical,
falsification, research_questions) — each self-contained, not wired into the
RAG pipeline directly. See each file's docstring for its specific model.

## `web/` — the browser UI

| File | Purpose |
|---|---|
| `app.py` | FastAPI app: `/`, `/api/query`, `/api/discover`, `/api/analyze`, `/api/fast`, `/api/stats`, `/api/papers`, `/api/ledger`, `/api/upload` |
| `templates/index.html` | Single-page UI: Query/Library/Ledger/Data Sources tabs |

## `interface/` — optional voice/avatar kiosk (`acheron interface`, separate from the web UI)

| File | Purpose |
|---|---|
| `nexus.py` | Session memory / predictive behavioral context |
| `ws_server.py` | WebSocket server for the kiosk frontend |
| `voice/stt.py`, `voice/tts.py` | Whisper speech-to-text, Piper/ElevenLabs text-to-speech |
| `avatar/renderer.py` | Avatar state + lip-sync data |

## Top-level directories outside `src/`

| Path | Purpose |
|---|---|
| `nexus_ingest/` | Enhanced PMC/PubMed full-text ingestion used by `collectors/pubmed.py` |
| `simulations/` | Standalone bioelectric/circuit simulation scripts (Acheron-side, not Nexus's RAG) |
| `scripts/` | One-off utility scripts, including `validate_parameters.py` (per-organism cited-vs-UNKNOWN report for the extracted parameter corpus) |
| `tests/` | Pytest suite — one file per module/feature, offline fixture-based (no live network calls) |
| `TROUBLESHOOTING.md` | Known Compute-layer / local-environment failure modes (connection errors, key issues, dependency version mismatches) and how to tell them apart |

## Adding a new collector (checklist)

1. Confirm the source's real API shape first — read its actual API docs or
   fetch its OpenAPI/swagger spec; never guess endpoint paths or field
   names (see `MANIFEST.md`'s "NO NUMERIC INVENTION RULE").
2. Subclass `BaseCollector`, implement `search(query, max_results) -> list[Paper]`.
3. Convert each record into a `Paper` whose `abstract` is built only from
   fields the source actually returned; set `organism`/`source_type` if
   the record isn't from general literature.
4. Add the source to `PaperSource` in `models.py`.
5. Wire it into `cli.py`'s `collect` command: add to the `--source` choice
   list, add an import, add a default-topics branch if the source needs
   different default queries than `GENE_PROTEIN_TOPICS`.
6. Add offline fixture-based parsing tests (see `tests/test_bio_collectors.py`,
   `tests/test_subtiwiki_collector.py`, and `tests/test_planmine_collector.py`
   for the pattern) plus, if the source represents a genuinely new record
   shape, a small end-to-end ChromaDB ingestion test.
