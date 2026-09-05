# Acheron Nexus

Domain-specific RAG assistant for **bioelectricity and biomedical research**. Collects papers from PubMed, bioRxiv, arXiv, and PhysioNet, and grounds itself further in curated protein, structure, and genomic data from UniProt, RCSB PDB, the AlphaFold Protein Structure Database, and NCBI Gene — then indexes everything into a vector store and answers natural-language queries with grounded, cited responses.

## Focus Areas

- Planarian bioelectricity and regeneration
- EEG cognitive patterns and brain oscillations
- Ion channel dynamics and electrophysiology
- Bioelectric morphogenesis and pattern formation
- Memory mechanisms in regenerating tissue
- Bioelectric computing

## Data Sources

| Source | Kind | What it adds |
|---|---|---|
| PubMed / PMC | Literature | Peer-reviewed papers, full text where PMC has it |
| bioRxiv / medRxiv | Literature | Preprints, not yet peer reviewed |
| arXiv (q-bio) | Literature | Quantitative biology / biophysics preprints |
| PhysioNet | Datasets | Published EEG/ECG and related signal datasets |
| [UniProtKB](https://www.uniprot.org) | Protein | Curated function, sequence, cross-species orthologs for innexins, connexins, ion channels |
| [RCSB PDB](https://www.rcsb.org) | Structure | Experimentally solved 3D structures (method, resolution) |
| [AlphaFold DB](https://alphafold.ebi.ac.uk) | Structure | Free DeepMind/EMBL-EBI predicted structures + per-model confidence (pLDDT), for proteins without a solved structure |
| [NCBI Gene](https://www.ncbi.nlm.nih.gov/gene) | Genomic | Chromosome location, RefSeq accessions, curated gene summaries |

All four bio-database collectors are free and require no API key (an `NCBI_API_KEY` raises NCBI Gene's rate limit but isn't required). They follow the project's science-first rule: every fact indexed comes from the source API's own response, never invented — see `MANIFEST.md`.

## Quickstart

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Edit .env with your API keys

# 3. Collect papers from all sources
acheron collect

# 4. Index into vector store
acheron index

# 5. Query (interactive mode)
acheron query -i

# 6. Or start the web UI
acheron serve
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `acheron collect` | Harvest papers from PubMed, bioRxiv, arXiv, PhysioNet |
| `acheron index` | Build/update the vector store from collected papers |
| `acheron query "question"` | Ask a question (single shot) |
| `acheron query -i` | Interactive query mode |
| `acheron query -r "question"` | Retrieve passages without LLM generation |
| `acheron add paper.pdf` | Add a local PDF manually |
| `acheron stats` | Show collection statistics |
| `acheron simulate --model grn --organism <name> --perturbation <gene>` | Simulate a gene knockdown's downstream effect using cited parameters extracted from the Library (`pip install -e ".[simulation]"` first) |
| `acheron serve` | Start the web interface |

The web interface (`acheron serve`) has four tabs: **Query** (ask + Discover/Analyze modes), **Library** (browse/search all indexed records with a per-source breakdown), **Ledger** (past discovery/analysis runs), and **Data Sources** (what each collector is and how to pull more of it in).

## Collection Options

```bash
# Collect from specific source
acheron collect --source pubmed

# Custom search queries
acheron collect -t "voltage-gated potassium channels" -t "gap junction signaling"

# More results per query
acheron collect -n 100

# Also download PDFs
acheron collect --download-pdfs
```

### Bio-database collection

The structural/genomic sources take gene or protein-style queries rather
than natural-language phrases, and aren't part of `--source all` (which
stays literature-only) since they need different query semantics:

```bash
# Protein function + sequence data (UniProtKB)
acheron collect --source uniprot -t "innexin" -t "connexin gap junction"

# Experimentally solved structures (RCSB PDB)
acheron collect --source pdb -t "connexin gap junction channel"

# Genomic context — chromosome location, RefSeq, curated summary (NCBI Gene)
acheron collect --source ncbi_gene -t "KCNQ1" -t "GJA1"

# Predicted structures (AlphaFold DB) — looked up by UniProt accession,
# so run --source uniprot first to discover accessions, then:
acheron collect --source alphafold -t "P17302,Q9Y6N1"

# Index everything (literature + bio-databases) into the vector store
acheron index
```

## Architecture

```
src/acheron/
├── cli.py                 # Click CLI interface
├── config.py              # Pydantic settings from .env
├── models.py              # Domain models (Paper, TextChunk, etc.)
├── collectors/
│   ├── base.py            # Base collector with HTTP + retry
│   ├── pubmed.py          # PubMed/PMC via NCBI E-Utilities
│   ├── biorxiv.py         # bioRxiv content API
│   ├── arxiv.py           # arXiv Atom feed API
│   ├── physionet.py       # PhysioNet dataset API
│   ├── uniprot.py         # UniProtKB protein function/sequence data
│   ├── structures.py      # RCSB PDB + AlphaFold DB structure data
│   └── ncbi_gene.py       # NCBI Gene genomic context
├── extraction/
│   ├── pdf_parser.py      # PDF → structured text (PyMuPDF + pdfplumber)
│   └── chunker.py         # Text → overlapping chunks
├── vectorstore/
│   └── store.py           # ChromaDB vector store
├── rag/
│   └── pipeline.py        # Retrieve → rerank → generate with citations
└── web/
    ├── app.py             # FastAPI application
    └── templates/
        └── index.html     # Web UI
```

## Configuration

All settings via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ACHERON_LLM_API_KEY` | (required) | OpenAI-compatible API key |
| `ACHERON_LLM_BASE_URL` | `https://api.openai.com/v1` | LLM endpoint |
| `ACHERON_LLM_MODEL` | `gpt-4o` | Model for generation |
| `ACHERON_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformer model |
| `ACHERON_EMBEDDING_API_KEY` | (empty) | If set, uses OpenAI embeddings instead |
| `NCBI_API_KEY` | (empty) | Optional; increases PubMed rate limits |
| `ACHERON_DATA_DIR` | `./data` | Storage root |
| `ACHERON_HOST` | `127.0.0.1` | Web server host |
| `ACHERON_PORT` | `8000` | Web server port |

## Embedding Options

**Local (default):** Uses `sentence-transformers/all-MiniLM-L6-v2` — no API key needed, runs on CPU.

**API-based:** Set `ACHERON_EMBEDDING_API_KEY` and `ACHERON_EMBEDDING_MODEL` to use OpenAI's `text-embedding-3-small` or similar.

## LLM Options

Supports any OpenAI-compatible endpoint:

- **OpenAI**: Set API key, use `gpt-4o` or `gpt-4o-mini`
- **Ollama**: Set `ACHERON_LLM_BASE_URL=http://localhost:11434/v1`, model to `llama3.1` or similar
- **vLLM / TGI**: Point to your local endpoint

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/
```

## License

MIT
