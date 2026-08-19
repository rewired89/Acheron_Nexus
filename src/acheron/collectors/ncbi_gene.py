"""Collector for NCBI Gene — genomic context for genes of interest.

Uses the same NCBI E-Utilities service as the PubMed collector
(db=gene instead of db=pubmed), so it shares the same API key / rate-limit
handling. Free, keyless (higher rate limit with NCBI_API_KEY set).

Gives Nexus real genomic facts — chromosome location, RefSeq accessions,
curated gene summaries, cross-species aliases — for genes such as
innexins (Drosophila/planarian gap junction genes), connexins (GJA1 etc.)
and voltage-gated ion channel genes (KCNQ, SCN, CACNA families).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from acheron.collectors.base import BaseCollector
from acheron.models import Paper, PaperSource

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


class NCBIGeneCollector(BaseCollector):
    """Harvest curated gene records from NCBI Gene."""

    source_name = "ncbi_gene"

    def search(self, query: str, max_results: int = 30) -> list[Paper]:
        ids = self._esearch(query, max_results)
        if not ids:
            logger.info("NCBI Gene: no results for '%s'", query)
            return []
        papers = self._esummary(ids)
        logger.info("NCBI Gene: found %d genes for '%s'", len(papers), query)
        return papers

    def _esearch(self, query: str, max_results: int) -> list[str]:
        params = {
            "db": "gene",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
        }
        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key
        try:
            resp = self._get(ESEARCH_URL, params=params)
            data = resp.json()
        except Exception:
            logger.warning("NCBI Gene esearch failed for '%s'", query)
            return []
        return data.get("esearchresult", {}).get("idlist", [])

    def _esummary(self, gene_ids: list[str]) -> list[Paper]:
        papers: list[Paper] = []
        batch_size = 100
        for start in range(0, len(gene_ids), batch_size):
            batch = gene_ids[start : start + batch_size]
            params = {
                "db": "gene",
                "id": ",".join(batch),
                "retmode": "json",
            }
            if self.settings.ncbi_api_key:
                params["api_key"] = self.settings.ncbi_api_key
            try:
                resp = self._get(ESUMMARY_URL, params=params)
                data = resp.json()
            except Exception:
                logger.warning("NCBI Gene esummary failed for batch starting at %d", start)
                continue

            result = data.get("result", {})
            for uid in result.get("uids", []):
                record = result.get(uid, {})
                paper = self._parse_record(uid, record)
                if paper:
                    papers.append(paper)

            time.sleep(0.15 if self.settings.ncbi_api_key else 0.4)
        return papers

    def _parse_record(self, uid: str, record: dict) -> Optional[Paper]:
        name = record.get("name", "")
        if not name and not uid:
            return None
        description = record.get("description", "")
        organism = record.get("organism", {}).get("scientificname", "")
        chromosome = record.get("chromosome", "")
        maplocation = record.get("maplocation", "")
        summary = record.get("summary", "")
        aliases = record.get("otheraliases", "")
        gene_type = record.get("genetype", "")

        genomicinfo = record.get("genomicinfo") or []
        accession = ""
        if genomicinfo:
            accession = genomicinfo[0].get("chraccver", "")

        summary_lines = [
            f"Gene: {name} — {description or 'no description'}",
            f"NCBI Gene ID: {uid}",
            f"Organism: {organism or 'unknown'}",
            f"Type: {gene_type or 'unknown'}",
            f"Location: chromosome {chromosome or '?'}"
            + (f" ({maplocation})" if maplocation else "")
            + (f", RefSeq {accession}" if accession else ""),
        ]
        if aliases:
            summary_lines.append(f"Aliases: {aliases}")
        if summary:
            summary_lines.append(f"Summary: {summary}")

        return Paper(
            paper_id=f"ncbigene:{uid}",
            title=f"{name} ({organism}) — NCBI Gene {uid}",
            authors=["NCBI"],
            abstract="\n".join(summary_lines),
            source=PaperSource.NCBI_GENE,
            journal="NCBI Gene",
            keywords=[a.strip() for a in aliases.split(",") if a.strip()] if aliases else [],
            url=f"https://www.ncbi.nlm.nih.gov/gene/{uid}",
        )
