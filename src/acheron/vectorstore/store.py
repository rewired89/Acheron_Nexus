"""ChromaDB-backed vector store for paper chunks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from acheron.config import get_settings
from acheron.models import Paper, QueryResult, TextChunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "acheron_papers"


class VectorStore:
    """Manages embedding storage and semantic retrieval via ChromaDB.

    Uses sentence-transformers for local embeddings by default,
    or an OpenAI-compatible API if configured.
    """

    def __init__(self, persist_dir: Optional[Path] = None) -> None:
        settings = get_settings()
        self.persist_dir = persist_dir or settings.vectorstore_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._embedding_fn = self._build_embedding_function(settings)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready — %d documents in collection", self.collection.count()
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def add_chunks(self, chunks: list[TextChunk]) -> int:
        """Add text chunks to the store. Returns the number added."""
        if not chunks:
            return 0

        # Deduplicate by chunk_id
        existing = set()
        try:
            existing_ids = self.collection.get(ids=[c.chunk_id for c in chunks])["ids"]
            existing = set(existing_ids)
        except Exception:
            pass

        new_chunks = [c for c in chunks if c.chunk_id not in existing]
        if not new_chunks:
            logger.debug("All %d chunks already indexed", len(chunks))
            return 0

        # ChromaDB has batch limits (~5000); we batch at 500
        batch_size = 500
        added = 0
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            self.collection.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "paper_id": c.paper_id,
                        "section": c.section,
                        "chunk_index": c.chunk_index,
                        "title": c.metadata.get("title", ""),
                        "authors": json.dumps(c.metadata.get("authors", [])),
                        "doi": c.metadata.get("doi", ""),
                        "pmid": c.metadata.get("pmid", ""),
                        "pmcid": c.metadata.get("pmcid", ""),
                        "source": c.metadata.get("source", ""),
                        "date": c.metadata.get("date", ""),
                        "url": c.metadata.get("url", ""),
                        "organism": c.organism or "",
                        "source_type": c.source_type or "",
                        # Evidence span fields
                        "source_file": c.source_file,
                        "span_start": c.span_start,
                        "span_end": c.span_end,
                        "excerpt": c.excerpt[:500] if c.excerpt else "",
                        "xpath": c.xpath,
                    }
                    for c in batch
                ],
            )
            added += len(batch)

        logger.info("Indexed %d new chunks (total: %d)", added, self.collection.count())
        return added

    def add_paper(self, paper: Paper, chunks: list[TextChunk]) -> int:
        """Index a paper's chunks. Also stores paper metadata for retrieval."""
        return self.add_chunks(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        n_results: int = 10,
        filter_source: Optional[str] = None,
    ) -> list[QueryResult]:
        """Semantic search returning ranked results with metadata."""
        where_filter = None
        if filter_source:
            where_filter = {"source": filter_source}

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count() or 1),
                where=where_filter,
            )
        except Exception:
            logger.exception("Vector search failed")
            return []

        query_results: list[QueryResult] = []
        if not results or not results["documents"]:
            return query_results

        docs = results["documents"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)

        for doc, meta, dist in zip(docs, metas, distances):
            authors_raw = meta.get("authors", "[]")
            try:
                authors = json.loads(authors_raw) if isinstance(authors_raw, str) else authors_raw
            except json.JSONDecodeError:
                authors = [authors_raw] if authors_raw else []

            # ChromaDB cosine distance → similarity
            relevance = 1.0 - dist

            query_results.append(QueryResult(
                text=doc,
                paper_id=meta.get("paper_id", ""),
                paper_title=meta.get("title", ""),
                authors=authors if isinstance(authors, list) else [str(authors)],
                doi=meta.get("doi", "") or None,
                pmid=meta.get("pmid", "") or None,
                pmcid=meta.get("pmcid", "") or None,
                section=meta.get("section", ""),
                relevance_score=relevance,
                url=meta.get("url", ""),
                source=meta.get("source", ""),
                # Evidence span fields
                source_file=meta.get("source_file", ""),
                span_start=meta.get("span_start", 0),
                span_end=meta.get("span_end", 0),
                excerpt=meta.get("excerpt", ""),
                xpath=meta.get("xpath", ""),
            ))

        return query_results

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------
    def count(self) -> int:
        return self.collection.count()

    def delete_paper(self, paper_id: str) -> int:
        """Remove all chunks for a given paper."""
        try:
            results = self.collection.get(where={"paper_id": paper_id})
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info("Deleted %d chunks for paper %s", len(results["ids"]), paper_id)
                return len(results["ids"])
        except Exception:
            logger.exception("Failed to delete paper %s", paper_id)
        return 0

    def list_papers(self) -> list[dict]:
        """List all unique papers in the store."""
        try:
            all_meta = self.collection.get()["metadatas"]
        except Exception:
            return []

        seen = {}
        for meta in all_meta:
            pid = meta.get("paper_id", "")
            if pid and pid not in seen:
                seen[pid] = {
                    "paper_id": pid,
                    "title": meta.get("title", ""),
                    "doi": meta.get("doi", ""),
                    "source": meta.get("source", ""),
                    "url": meta.get("url", ""),
                    "pmid": meta.get("pmid", ""),
                }
        return list(seen.values())

    # ------------------------------------------------------------------
    # Embedding function
    # ------------------------------------------------------------------
    @staticmethod
    def _build_embedding_function(settings):
        """Build the ChromaDB-compatible embedding function."""
        if settings.embedding_api_key:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
            return OpenAIEmbeddingFunction(
                api_key=settings.embedding_api_key,
                model_name=settings.embedding_model,
            )
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            return SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model,
            )
        except Exception:
            logger.warning(
                "SentenceTransformer model download failed, "
                "falling back to ChromaDB default embedding function"
            )
            return chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
