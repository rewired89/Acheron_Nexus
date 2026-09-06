"""Collectors for protein structure data — RCSB PDB and the AlphaFold DB.

Both are free, keyless APIs. This is the practical version of "use
AlphaFold-quality data": rather than training a new structure-prediction
model, Nexus pulls the real, freely published structures (experimental
from RCSB PDB, predicted from DeepMind/EMBL-EBI's AlphaFold DB) for the
ion-channel and gap-junction proteins central to Acheron's bioelectricity
work, and indexes them as grounded, citable records.

RCSB Search API:  https://search.rcsb.org/rcsbsearch/v2/query
RCSB Data API:    https://data.rcsb.org/rest/v1/core/entry/{pdb_id}
AlphaFold DB API: https://alphafold.ebi.ac.uk/api/prediction/{uniprot_accession}
"""

from __future__ import annotations

import logging
from typing import Optional

from acheron.collectors.base import BaseCollector
from acheron.models import Paper, PaperSource, SourceType

logger = logging.getLogger(__name__)

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
ALPHAFOLD_PREDICTION_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"


class PDBCollector(BaseCollector):
    """Harvest experimentally determined structures from RCSB PDB."""

    source_name = "pdb"

    def search(self, query: str, max_results: int = 25) -> list[Paper]:
        """Full-text search RCSB PDB, then fetch entry details for each hit."""
        ids = self._search_ids(query, max_results)
        if not ids:
            logger.info("PDB: no results for '%s'", query)
            return []

        papers: list[Paper] = []
        for pdb_id in ids:
            paper = self._fetch_entry(pdb_id)
            if paper:
                papers.append(paper)
        logger.info("PDB: found %d entries for '%s'", len(papers), query)
        return papers

    def _search_ids(self, query: str, max_results: int) -> list[str]:
        body = {
            "query": {
                "type": "terminal",
                "service": "full_text",
                "parameters": {"value": query},
            },
            "return_type": "entry",
            "request_options": {"paginate": {"rows": min(max_results, 100)}},
        }
        try:
            resp = self.client.post(RCSB_SEARCH_URL, json=body, timeout=60.0)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.warning("RCSB search failed for query '%s'", query)
            return []
        return [r.get("identifier", "") for r in data.get("result_set", []) if r.get("identifier")]

    def _fetch_entry(self, pdb_id: str) -> Optional[Paper]:
        try:
            resp = self._get(RCSB_ENTRY_URL.format(pdb_id=pdb_id))
            entry = resp.json()
        except Exception:
            logger.debug("Failed to fetch RCSB entry %s", pdb_id)
            return None
        return self._parse_entry(pdb_id, entry)

    def _parse_entry(self, pdb_id: str, entry: dict) -> Optional[Paper]:
        title = entry.get("struct", {}).get("title", pdb_id)
        methods = [m.get("method", "") for m in entry.get("exptl", []) if m.get("method")]
        resolutions = entry.get("rcsb_entry_info", {}).get("resolution_combined") or []
        keywords_field = entry.get("struct_keywords", {}).get("pdbx_keywords", "")
        deposit_date = entry.get("rcsb_accession_info", {}).get("deposit_date", "")

        summary_lines = [
            f"PDB entry {pdb_id}: {title}",
            f"Experimental method: {', '.join(methods) if methods else 'unknown'}",
        ]
        if resolutions:
            summary_lines.append(
                "Resolution: " + ", ".join(f"{r} Angstrom" for r in resolutions)
            )
        if keywords_field:
            summary_lines.append(f"Classification: {keywords_field}")

        pub_date = None
        if deposit_date:
            try:
                from datetime import date as _date

                pub_date = _date.fromisoformat(deposit_date[:10])
            except ValueError:
                pass

        return Paper(
            paper_id=f"pdb:{pdb_id}",
            title=f"PDB {pdb_id} — {title}",
            authors=["RCSB PDB"],
            abstract="\n".join(summary_lines),
            publication_date=pub_date,
            source=PaperSource.PDB,
            journal="RCSB Protein Data Bank",
            keywords=[keywords_field] if keywords_field else [],
            url=f"https://www.rcsb.org/structure/{pdb_id}",
        )


class AlphaFoldCollector(BaseCollector):
    """Fetch predicted structure models from the AlphaFold Protein Structure DB.

    Unlike the other collectors this is a direct lookup by UniProt
    accession rather than a keyword search — pass the accession(s) as a
    comma-separated `query`, e.g. "P17302,P48745" (connexins / innexins).
    Combine with UniProtCollector to discover accessions for a gene
    family first.
    """

    source_name = "alphafold"

    def search(self, query: str, max_results: int = 50) -> list[Paper]:
        accessions = [a.strip() for a in query.replace(" ", ",").split(",") if a.strip()]
        papers: list[Paper] = []
        for accession in accessions[:max_results]:
            paper = self._fetch_prediction(accession)
            if paper:
                papers.append(paper)
        logger.info("AlphaFold DB: found %d predictions for '%s'", len(papers), query)
        return papers

    def _fetch_prediction(self, accession: str) -> Optional[Paper]:
        try:
            resp = self._get(ALPHAFOLD_PREDICTION_URL.format(accession=accession))
            data = resp.json()
        except Exception:
            logger.debug("No AlphaFold prediction for accession %s", accession)
            return None
        return self._parse_prediction(accession, data)

    def _parse_prediction(self, accession: str, data) -> Optional[Paper]:
        if not data:
            return None

        record = data[0] if isinstance(data, list) else data
        entry_id = record.get("entryId", f"AF-{accession}-F1")
        gene = record.get("gene", "")
        organism = record.get("organismScientificName", "")
        confidence = record.get("globalMetricValue")
        model_version = record.get("latestVersion")
        model_url = record.get("pdbUrl") or record.get("cifUrl", "")
        created = record.get("modelCreatedDate", "")

        summary_lines = [
            f"AlphaFold predicted structure {entry_id} for UniProt {accession}"
            + (f" ({gene})" if gene else ""),
            f"Organism: {organism or 'unknown'}",
        ]
        if confidence is not None:
            summary_lines.append(
                f"Global confidence (mean pLDDT): {confidence:.1f}/100 "
                "(>90 very high, 70-90 confident, <50 low confidence — "
                "treat low-confidence regions as unresolved, not fact)"
            )
        summary_lines.append(f"Model version: v{model_version}" if model_version else "")
        summary_lines.append(f"Model file: {model_url}" if model_url else "")

        pub_date = None
        if created:
            try:
                from datetime import date as _date

                pub_date = _date.fromisoformat(created[:10])
            except ValueError:
                pass

        return Paper(
            paper_id=f"alphafold:{entry_id}",
            title=f"AlphaFold model {entry_id} ({gene or accession})",
            authors=["DeepMind", "EMBL-EBI"],
            abstract="\n".join(line for line in summary_lines if line),
            publication_date=pub_date,
            source=PaperSource.ALPHAFOLD,
            source_type=SourceType.CURATED_DB,
            organism=organism,
            journal="AlphaFold Protein Structure Database",
            keywords=[gene] if gene else [],
            url=f"https://alphafold.ebi.ac.uk/entry/{accession}",
        )
