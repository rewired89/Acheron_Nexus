"""Collector for UniProtKB — curated protein sequence and function data.

Free, keyless REST API (https://rest.uniprot.org). Used to ground Nexus's
reasoning in real protein-level facts (function, domains, cross-species
orthologs, cross-references to structures) for the gene families Acheron
cares about: innexins, connexins, voltage-gated ion channels, etc.

Each UniProt entry is converted into a Paper whose "abstract" is a
structured plain-text summary of the curated annotation (function,
organism, length, subcellular location, keywords) so it flows through the
existing chunker / vector store / RAG pipeline unchanged. Numeric facts
(sequence length, cross-reference IDs) come directly from the API
response — never invented — per the project's science-first rules.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from acheron.collectors.base import BaseCollector
from acheron.models import Paper, PaperSource, SourceType

logger = logging.getLogger(__name__)

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

FIELDS = ",".join(
    [
        "accession",
        "id",
        "protein_name",
        "gene_names",
        "organism_name",
        "length",
        "cc_function",
        "cc_subcellular_location",
        "keyword",
        "xref_pdb",
        "date_modified",
    ]
)


class UniProtCollector(BaseCollector):
    """Harvest curated protein records from UniProtKB."""

    source_name = "uniprot"

    def search(self, query: str, max_results: int = 50) -> list[Paper]:
        """Search UniProtKB (reviewed/Swiss-Prot entries preferred).

        `query` is a free-text or field query, e.g. "innexin", "connexin",
        "gene:KCNQ1", "voltage-gated potassium channel AND reviewed:true".
        """
        params = {
            "query": query,
            "fields": FIELDS,
            "format": "json",
            "size": min(max_results, 500),
        }
        try:
            resp = self._get(UNIPROT_SEARCH_URL, params=params)
            data = resp.json()
        except Exception:
            logger.warning("UniProt API request failed for query '%s'", query)
            return []

        papers: list[Paper] = []
        for entry in data.get("results", [])[:max_results]:
            paper = self._parse_entry(entry)
            if paper:
                papers.append(paper)

        logger.info("UniProt: found %d entries for '%s'", len(papers), query)
        return papers

    def _parse_entry(self, entry: dict) -> Optional[Paper]:
        accession = entry.get("primaryAccession", "")
        if not accession:
            return None
        entry_name = entry.get("uniProtkbId", "")

        protein_desc = entry.get("proteinDescription", {})
        protein_name = (
            protein_desc.get("recommendedName", {}).get("fullName", {}).get("value", "")
            or (protein_desc.get("submissionNames") or [{}])[0]
            .get("fullName", {})
            .get("value", "")
            or entry_name
            or accession
        )

        genes = entry.get("genes", [])
        gene_names = [g.get("geneName", {}).get("value", "") for g in genes if g.get("geneName")]
        gene_names = [g for g in gene_names if g]

        organism = entry.get("organism", {}).get("scientificName", "")
        taxon_id = entry.get("organism", {}).get("taxonId")

        seq_length = entry.get("sequence", {}).get("length")

        function_texts = []
        location_texts = []
        for comment in entry.get("comments", []):
            ctype = comment.get("commentType", "")
            texts = [t.get("value", "") for t in comment.get("texts", []) if t.get("value")]
            if ctype == "FUNCTION":
                function_texts.extend(texts)
            elif ctype == "SUBCELLULAR LOCATION":
                for loc in comment.get("subcellularLocations", []):
                    val = loc.get("location", {}).get("value", "")
                    if val:
                        location_texts.append(val)
                location_texts.extend(texts)

        keywords = [kw.get("name", "") for kw in entry.get("keywords", []) if kw.get("name")]

        pdb_ids = [
            xref.get("id", "")
            for xref in entry.get("uniProtKBCrossReferences", [])
            if xref.get("database") == "PDB" and xref.get("id")
        ]

        modified = entry.get("entryAudit", {}).get("lastAnnotationUpdateDate") or entry.get(
            "date_modified"
        )
        pub_date = None
        if isinstance(modified, str) and len(modified) >= 10:
            try:
                pub_date = date.fromisoformat(modified[:10])
            except ValueError:
                pass

        summary_lines = [
            f"Protein: {protein_name}",
            f"UniProt accession: {accession} ({entry_name})",
            f"Gene(s): {', '.join(gene_names) if gene_names else 'unknown'}",
            f"Organism: {organism or 'unknown'}"
            + (f" (NCBI taxon {taxon_id})" if taxon_id else ""),
            f"Sequence length: {seq_length} aa" if seq_length else "Sequence length: unknown",
        ]
        if function_texts:
            summary_lines.append("Function: " + " ".join(function_texts))
        if location_texts:
            summary_lines.append("Subcellular location: " + "; ".join(sorted(set(location_texts))))
        if keywords:
            summary_lines.append("Keywords: " + ", ".join(keywords))
        if pdb_ids:
            summary_lines.append(
                "Experimentally determined structures (PDB): " + ", ".join(pdb_ids)
            )
        else:
            summary_lines.append(
                "No experimentally determined structure in PDB — "
                "see AlphaFold DB for a predicted model."
            )

        abstract = "\n".join(summary_lines)

        return Paper(
            paper_id=f"uniprot:{accession}",
            title=f"{protein_name} — UniProtKB {accession}",
            authors=["UniProt Consortium"],
            abstract=abstract,
            publication_date=pub_date,
            source=PaperSource.UNIPROT,
            source_type=SourceType.CURATED_DB,
            organism=organism,
            journal="UniProtKB/Swiss-Prot",
            keywords=keywords + gene_names,
            url=f"https://www.uniprot.org/uniprotkb/{accession}/entry",
        )
