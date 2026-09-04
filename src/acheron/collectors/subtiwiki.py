"""Collector for SubtiWiki — the curated functional genomics database for
Bacillus subtilis (subtiwiki.uni-goettingen.de).

Uses SubtiWiki's public, keyless REST API (base URL confirmed from its own
Swagger/OpenAPI spec at https://subtiwiki.uni-goettingen.de/v5/api/swagger.json,
fetched 2026-09-04). Every field this collector reads is a documented field
in that spec; nothing about the API shape is guessed. Per the project's
science-first rule (see MANIFEST.md / LOGIC_HASH.md: "NO NUMERIC INVENTION
RULE"), only values SubtiWiki actually returns are included — if a field is
absent for a given gene, it is simply omitted from the summary, never
filled in or inferred.

Each SubtiWiki gene record is converted into a Paper (organism="B. subtilis",
source_type=SourceType.CURATED_DB) whose "abstract" is a structured
plain-text summary — locus tag, function, protein-protein interactions,
regulon/transcription-factor relationships, and expression-by-condition
data — so it flows through the existing chunker / vector store / RAG
pipeline unchanged, the same pattern used by uniprot.py / structures.py.

API response envelope: every SubtiWiki endpoint wraps its payload as
{"code": ..., "is_success": ..., "data": <actual payload>, "message": ...}
— this collector unwraps ["data"] before parsing.

NOT YET CONFIRMED: the human-readable gene page URL pattern on the
SubtiWiki website (the swagger spec only documents the API, not the
frontend's routing). `Paper.url` below points at the real, verified API
endpoint for the gene instead of guessing a page URL.
"""

from __future__ import annotations

import logging
from typing import Optional

from acheron.collectors.base import BaseCollector
from acheron.models import Paper, PaperSource, SourceType

logger = logging.getLogger(__name__)

BASE_URL = "https://subtiwiki.uni-goettingen.de/v5/api"
SEARCH_URL = f"{BASE_URL}/search/"
GENE_URL = f"{BASE_URL}/gene/{{id}}"
EXPRESSION_URL = f"{BASE_URL}/expression/"

ORGANISM = "B. subtilis"

# Regulation "mechanism" values as documented in the GeneRegulation schema
# (used only to render existing values readably; never invented).
_MECHANISM_LABELS = {
    "Transcriptional": "transcriptional",
    "Translational": "translational",
    "TranscriptionFactor": "transcription factor",
    "Attenuation": "attenuation",
    "TerminationAntitermination": "termination/antitermination",
    "RnaSwitch": "RNA switch",
    "SigmaFactor": "sigma factor",
    "AntiSenseRna": "antisense RNA",
    "Indirect": "indirect",
}

MAX_EXPRESSION_CONDITIONS_SHOWN = 20


class SubtiWikiCollector(BaseCollector):
    """Harvest curated gene/protein/regulatory records from SubtiWiki."""

    source_name = "subtiwiki"

    def search(self, query: str, max_results: int = 50) -> list[Paper]:
        """Search SubtiWiki for genes matching `query` (a gene name or
        locus tag, e.g. "sigB" or "BSU_00010") and return one Paper per
        matching gene, each carrying its full available record.
        """
        gene_ids = self._search_gene_ids(query, max_results)
        papers: list[Paper] = []
        for gene_id in gene_ids:
            paper = self._fetch_gene_paper(gene_id)
            if paper:
                papers.append(paper)

        logger.info("SubtiWiki: found %d gene records for '%s'", len(papers), query)
        return papers

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------
    def _unwrap(self, resp) -> Optional[dict]:
        """Unwrap SubtiWiki's {"code","is_success","data","message"} envelope."""
        try:
            body = resp.json()
        except Exception:
            return None
        if not body.get("is_success", True):
            logger.warning("SubtiWiki API reported failure: %s", body.get("message"))
            return None
        return body.get("data")

    def _search_gene_ids(self, query: str, max_results: int) -> list[int]:
        params = {"q": query, "category": "Gene", "mode": "all", "limit": max_results}
        try:
            resp = self._get(SEARCH_URL, params=params)
        except Exception:
            logger.warning("SubtiWiki search request failed for '%s'", query)
            return []

        data = self._unwrap(resp)
        if not data:
            return []

        hits = list(data.get("exact_hits", [])) + list(data.get("partial_hits", []))
        ids: list[int] = []
        seen = set()
        for hit in hits:
            gene_id = hit.get("id")
            if gene_id is None or gene_id in seen:
                continue
            seen.add(gene_id)
            ids.append(gene_id)
            if len(ids) >= max_results:
                break
        return ids

    def _fetch_gene_paper(self, gene_id: int) -> Optional[Paper]:
        try:
            resp = self._get(GENE_URL.format(id=gene_id), params={"representation": "default"})
        except Exception:
            logger.warning("SubtiWiki: failed to fetch gene %s", gene_id)
            return None

        gene = self._unwrap(resp)
        if not gene:
            return None

        expression = self._fetch_expression(gene_id)
        return self._parse_gene(gene, expression)

    def _fetch_expression(self, gene_id: int) -> list[dict]:
        try:
            resp = self._get(EXPRESSION_URL, params={"geneId": gene_id})
        except Exception:
            logger.debug("SubtiWiki: no expression data fetched for gene %s", gene_id)
            return []
        data = self._unwrap(resp)
        return data if isinstance(data, list) else []

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _parse_gene(self, gene: dict, expression: list[dict]) -> Optional[Paper]:
        gene_id = gene.get("id")
        name = gene.get("name")
        if gene_id is None or not name:
            return None

        locus_tag = ""
        genomic_annotations = gene.get("genomic_annotations") or []
        if genomic_annotations:
            locus_tag = genomic_annotations[0].get("locus_tag", "") or ""

        summary_lines = [
            f"Gene: {name}",
            f"Locus tag: {locus_tag}" if locus_tag else "Locus tag: not recorded in SubtiWiki",
        ]

        if gene.get("function"):
            summary_lines.append(f"Function: {gene['function']}")
        if gene.get("description"):
            summary_lines.append(f"Description: {gene['description']}")
        if gene.get("product"):
            summary_lines.append(f"Product: {gene['product']}")
        if gene.get("essential"):
            summary_lines.append(f"Essential: {gene['essential']}")

        mutant_phenotypes = gene.get("mutant_phenotypes") or []
        if mutant_phenotypes:
            summary_lines.append("Mutant phenotypes: " + "; ".join(mutant_phenotypes))

        # --- Protein-protein interactions (Gene.protein.interactions) ---
        protein = gene.get("protein") or {}
        interactions = protein.get("interactions") or []
        if interactions:
            interaction_lines = []
            for interaction in interactions:
                partner_names = [
                    m.get("name", "") for m in (interaction.get("molecules") or []) if m.get("name")
                ]
                partner_names = [p for p in partner_names if p != name]
                if not partner_names:
                    continue
                line = f"Interacts with {', '.join(partner_names)}"
                if interaction.get("description"):
                    line += f" ({interaction['description']})"
                interaction_lines.append(line)
            if interaction_lines:
                summary_lines.append("Protein-protein interactions:")
                summary_lines.extend(f"  - {line}" for line in interaction_lines)

        # --- Regulon / transcription-factor relationships ---
        regulations = gene.get("regulations") or []
        if regulations:
            reg_lines = []
            for reg in regulations:
                regulon = reg.get("regulon") or {}
                regulator_name = (
                    regulon.get("regulator_display_name")
                    or (regulon.get("regulator_gene") or {}).get("name")
                    or "unnamed regulator"
                )
                mechanism = _MECHANISM_LABELS.get(reg.get("mechanism", ""), reg.get("mechanism", ""))
                mode = reg.get("mode", "")
                line = f"Regulated by {regulator_name}"
                details = ", ".join(p for p in (mechanism, mode.lower() if mode else "") if p)
                if details:
                    line += f" ({details})"
                if reg.get("description"):
                    line += f": {reg['description']}"
                reg_lines.append(line)
            if reg_lines:
                summary_lines.append("Regulation:")
                summary_lines.extend(f"  - {line}" for line in reg_lines)

        own_regulon = gene.get("regulon")
        if own_regulon:
            n_genes = len(own_regulon.get("gene_regulations") or [])
            n_operons = len(own_regulon.get("operon_regulations") or [])
            if n_genes or n_operons:
                summary_lines.append(
                    f"Acts as a regulator: regulates {n_genes} gene(s) and "
                    f"{n_operons} operon(s) directly (per SubtiWiki's regulon record)."
                )

        expr_and_reg = gene.get("expression_and_regulation") or []
        if expr_and_reg:
            summary_lines.append("Expression and regulation notes: " + "; ".join(expr_and_reg))

        # --- Expression-by-condition data (separate /expression/ endpoint) ---
        if expression:
            shown = expression[:MAX_EXPRESSION_CONDITIONS_SHOWN]
            expr_lines = []
            for entry in shown:
                condition = entry.get("condition") or {}
                cond_name = condition.get("name", "unknown condition")
                value = entry.get("value")
                if value is not None:
                    expr_lines.append(f"{cond_name}: {value}")
            if expr_lines:
                summary_lines.append("Expression by condition:")
                summary_lines.extend(f"  - {line}" for line in expr_lines)
                remaining = len(expression) - len(shown)
                if remaining > 0:
                    summary_lines.append(f"  ({remaining} additional condition(s) not shown)")

        references = gene.get("references") or []
        pubmed_ids = [str(r["pubmed_id"]) for r in references if r.get("pubmed_id")]
        if pubmed_ids:
            summary_lines.append("Cited by SubtiWiki (PubMed IDs): " + ", ".join(pubmed_ids))

        abstract = "\n".join(summary_lines)

        return Paper(
            paper_id=f"subtiwiki:gene:{gene_id}",
            title=f"{name} ({locus_tag}), SubtiWiki gene record" if locus_tag else f"{name}, SubtiWiki gene record",
            authors=["SubtiWiki (University of Göttingen)"],
            abstract=abstract,
            source=PaperSource.SUBTIWIKI,
            journal="SubtiWiki",
            keywords=[name] + ([locus_tag] if locus_tag else []),
            url=GENE_URL.format(id=gene_id),
            organism=ORGANISM,
            source_type=SourceType.CURATED_DB,
        )
