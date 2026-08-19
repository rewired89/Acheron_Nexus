"""Split paper text into overlapping chunks for vector embedding.

Includes evidence span tracking (byte offsets, excerpts) for precise citations.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from acheron.models import Paper, TextChunk

logger = logging.getLogger(__name__)


def _make_excerpt(text: str, max_len: int = 300) -> str:
    """Create a short excerpt for evidence display."""
    if len(text) <= max_len:
        return text.strip()
    # Try to break at sentence boundary
    excerpt = text[:max_len]
    last_period = excerpt.rfind(". ")
    if last_period > max_len // 2:
        return excerpt[: last_period + 1].strip()
    return excerpt.strip() + "..."


class TextChunker:
    """Chunk paper text with configurable size, overlap, and boundary respect."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
    ) -> None:
        self.chunk_size = chunk_size  # in tokens (approx words)
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_paper(self, paper: Paper) -> list[TextChunk]:
        """Create text chunks from a paper, preserving section structure.

        Tracks byte offsets for evidence span citations.
        """
        chunks: list[TextChunk] = []
        chunk_index = 0

        # Determine source file for evidence tracking
        source_file = ""
        if paper.pdf_path:
            source_file = Path(paper.pdf_path).name
        elif paper.pmid:
            source_file = f"pubmed_{paper.pmid}.json"
        elif paper.pmcid:
            source_file = f"pmc_{paper.pmcid}.nxml"

        # If we have structured sections, chunk by section
        if paper.sections:
            for section in paper.sections:
                if not section.text.strip():
                    continue
                section_chunks, offsets = self._chunk_text_with_offsets(section.text)
                for text, (start, end) in zip(section_chunks, offsets):
                    if section.heading:
                        xpath = f"section[@heading='{section.heading}']"
                    else:
                        xpath = "section"
                    chunks.append(self._make_chunk(
                        paper=paper,
                        text=text,
                        section=section.heading,
                        chunk_index=chunk_index,
                        span_start=start,
                        span_end=end,
                        source_file=source_file,
                        xpath=xpath,
                    ))
                    chunk_index += 1

        # Also chunk the abstract separately (high-value content)
        if paper.abstract:
            chunks.insert(0, self._make_chunk(
                paper=paper,
                text=paper.abstract,
                section="Abstract",
                chunk_index=-1,
                span_start=0,
                span_end=len(paper.abstract),
                source_file=source_file,
                xpath="abstract",
            ))

        # If no sections but we have full_text, chunk the whole thing
        if not paper.sections and paper.full_text:
            full_chunks, offsets = self._chunk_text_with_offsets(paper.full_text)
            for text, (start, end) in zip(full_chunks, offsets):
                chunks.append(self._make_chunk(
                    paper=paper,
                    text=text,
                    section="",
                    chunk_index=chunk_index,
                    span_start=start,
                    span_end=end,
                    source_file=source_file,
                ))
                chunk_index += 1

        # Include tables as separate chunks
        for i, table in enumerate(paper.tables):
            table_text = table.to_text()
            if table_text.strip():
                chunks.append(self._make_chunk(
                    paper=paper,
                    text=table_text,
                    section=f"Table {i + 1}",
                    chunk_index=chunk_index,
                    source_file=source_file,
                    xpath=f"table[{i + 1}]",
                ))
                chunk_index += 1

        logger.debug(
            "Chunked paper '%s' into %d chunks", paper.title[:60], len(chunks)
        )
        return chunks

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks respecting sentence boundaries."""
        chunks, _ = self._chunk_text_with_offsets(text)
        return chunks

    def _chunk_text_with_offsets(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        """Split text into overlapping chunks, returning text and (start, end) offsets."""
        sentences = self._split_sentences(text)
        if not sentences:
            return [], []

        chunks: list[str] = []
        offsets: list[tuple[int, int]] = []
        current_words: list[str] = []
        current_len = 0

        # Track position in original text
        pos = 0
        sentence_positions: list[tuple[int, int]] = []
        for sentence in sentences:
            # Find sentence in original text
            idx = text.find(sentence, pos)
            if idx == -1:
                idx = pos
            end = idx + len(sentence)
            sentence_positions.append((idx, end))
            pos = end

        sent_idx = 0
        chunk_start_sent = 0

        for sentence in sentences:
            words = sentence.split()
            sent_len = len(words)

            if current_len + sent_len > self.chunk_size and current_len > 0:
                chunk_text = " ".join(current_words)
                if len(current_words) >= self.min_chunk_size:
                    chunks.append(chunk_text.strip())
                    # Calculate offsets
                    start = sentence_positions[chunk_start_sent][0]
                    end = sentence_positions[sent_idx - 1][1] if sent_idx > 0 else start
                    offsets.append((start, end))

                # Keep overlap (for next chunk, start will be adjusted)
                overlap_words = current_words[-self.chunk_overlap :] if self.chunk_overlap else []
                overlap_sent_count = 0
                temp_len = 0
                for i in range(len(current_words) - 1, -1, -1):
                    if temp_len >= self.chunk_overlap:
                        break
                    temp_len += 1
                    overlap_sent_count += 1

                current_words = overlap_words
                current_len = len(current_words)
                # Adjust start for next chunk (overlap region)
                chunk_start_sent = max(0, sent_idx - overlap_sent_count)

            current_words.extend(words)
            current_len += sent_len
            sent_idx += 1

        # Final chunk
        if current_words and len(current_words) >= self.min_chunk_size:
            chunks.append(" ".join(current_words).strip())
            if chunk_start_sent < len(sentence_positions):
                start = sentence_positions[chunk_start_sent][0]
            else:
                start = 0
            end = sentence_positions[-1][1] if sentence_positions else len(text)
            offsets.append((start, end))

        return chunks, offsets

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex heuristics."""
        # Handle common abbreviations to avoid false splits
        text = re.sub(r"(Dr|Mr|Mrs|Ms|Prof|Fig|Eq|Ref|Vol|No|et al)\.", r"\1<DOT>", text)
        text = re.sub(r"(\d)\.", r"\1<DOT>", text)  # decimal numbers

        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Restore dots
        return [s.replace("<DOT>", ".") for s in sentences if s.strip()]

    def _make_chunk(
        self,
        paper: Paper,
        text: str,
        section: str,
        chunk_index: int,
        span_start: int = 0,
        span_end: int = 0,
        source_file: str = "",
        xpath: str = "",
    ) -> TextChunk:
        content_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        chunk_id = f"{paper.paper_id}::{chunk_index}::{content_hash}"

        # Determine source file
        if not source_file and paper.pdf_path:
            source_file = Path(paper.pdf_path).name

        return TextChunk(
            chunk_id=chunk_id,
            paper_id=paper.paper_id,
            text=text,
            section=section,
            chunk_index=chunk_index,
            metadata={
                "title": paper.title,
                "authors": paper.authors,
                "doi": paper.doi or "",
                "pmid": paper.pmid or "",
                "pmcid": paper.pmcid or "",
                "source": paper.source,
                "date": str(paper.publication_date) if paper.publication_date else "",
                "url": paper.url or "",
            },
            # Evidence span fields
            source_file=source_file,
            span_start=span_start,
            span_end=span_end,
            excerpt=_make_excerpt(text),
            xpath=xpath,
        )
