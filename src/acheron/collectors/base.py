"""Base collector interface."""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Optional

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from acheron.config import get_settings
from acheron.models import Paper


def _is_retryable(exc: BaseException) -> bool:
    """True for transient failures worth retrying, false for 4xx client errors.

    A 404/403/etc. means the URL is wrong or the content isn't there --
    retrying it 3 times with exponential backoff just wastes minutes on a
    request guaranteed to fail identically every time (this is exactly what
    made `--download-pdfs` runs appear to hang: bioRxiv's guessed PDF URL
    pattern 404s for any preprint past v1, and each one burned ~3 minutes
    of pointless retries).
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, (httpx.TransportError, httpx.TimeoutException))


logger = logging.getLogger(__name__)


class BaseCollector(abc.ABC):
    """Abstract base for all paper collectors."""

    source_name: str = "base"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = httpx.Client(timeout=60.0, follow_redirects=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def search(self, query: str, max_results: int = 50) -> list[Paper]:
        """Search the source and return Paper objects with metadata."""
        ...

    def download_pdf(self, paper: Paper, target_dir: Optional[Path] = None) -> Optional[Path]:
        """Download the PDF for a paper, returning the local path."""
        if not paper.url:
            logger.warning("No URL for paper %s — skipping PDF download", paper.paper_id)
            return None

        target_dir = target_dir or self.settings.pdf_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_id = paper.paper_id.replace("/", "_").replace(":", "_")
        dest = target_dir / f"{safe_id}.pdf"

        if dest.exists():
            logger.debug("PDF already on disk: %s", dest)
            paper.pdf_path = str(dest)
            return dest

        try:
            return self._download(paper.url, dest, paper)
        except Exception:
            logger.exception("Failed to download PDF for %s", paper.paper_id)
            return None

    def save_metadata(self, paper: Paper, target_dir: Optional[Path] = None) -> Path:
        """Persist paper metadata as JSON."""
        target_dir = target_dir or self.settings.metadata_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_id = paper.paper_id.replace("/", "_").replace(":", "_")
        dest = target_dir / f"{safe_id}.json"
        dest.write_text(paper.model_dump_json(indent=2), encoding="utf-8")
        return dest

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, max=30),
        retry=retry_if_exception(_is_retryable),
    )
    def _get(self, url: str, **kwargs) -> httpx.Response:
        resp = self.client.get(url, **kwargs)
        resp.raise_for_status()
        return resp

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, max=30),
        retry=retry_if_exception(_is_retryable),
    )
    def _download(self, url: str, dest: Path, paper: Paper) -> Path:
        with self.client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=8192):
                    f.write(chunk)
        paper.pdf_path = str(dest)
        logger.info("Downloaded PDF → %s", dest.name)
        return dest

    def close(self) -> None:
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
