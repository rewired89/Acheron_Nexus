"""Collector for PlanMine — the genomics/transcriptomics database for the
planarian Schmidtea mediterranea (planmine.mpinat.mpg.de), built on the
InterMine data-warehouse platform.

FIELD NAMES CONFIRMED REAL, NOT GUESSED: every class/attribute referenced
below (Gene, Contig, Homologue, RNAiResult, DomainHit, ProteinDomain,
GOAnnotation, ...) is taken directly from PlanMine's own live schema
introspection endpoint, GET {base}/service/model, fetched 2026-09-04. This
sandbox's network policy blocks direct access to planmine.mpinat.mpg.de
(and to InterMine's own registry), so that model.json was fetched by the
user and pasted in rather than retrieved by this code -- the same
verification discipline as subtiwiki.py, just via a different channel.

NOT LIVE-TESTED (flagged honestly, per MANIFEST.md's no-invention rule --
guessing here would be exactly the mistake that rule exists to prevent):
InterMine's REST query wire format (PathQuery XML in, `format=json` rows
out) is a long-standing, publicly documented, cross-mine standard (the
same format used by FlyMine, MouseMine, RatMine, and implemented by the
official `intermine-ws-python` client), not something specific to
PlanMine -- but this collector has not been run against a live PlanMine
response, only against PlanMine's confirmed real schema. The response
*shape* (`{"results": [[col1, col2, ...], ...]}`, row order matching the
query's `view` list) is the standard InterMine convention; if PlanMine's
instance differs, `acheron collect --source planmine` will surface it
immediately as a parse warning rather than silently mis-parsing data.

WHY THIS DATA MATTERS DOWNSTREAM: ion channel / innexin (gap junction)
annotations feed Acheron's own BETSE-based bioelectric simulations (a
separate repo/module -- this collector only gathers the data, it does not
touch any simulation code, per this phase's scope).

======================================================================
DATA COMPLETENESS ASYMMETRY vs. SubtiWiki (Phase 1) -- READ BEFORE USING
THIS DATA IN ANY CONFIDENCE-SCORING LOGIC (Phase 4):
======================================================================
PlanMine's schema is sequence/annotation/expression-centric, not a
curated functional-genomics wiki like SubtiWiki. Concretely, per the
schema above and PlanMine's own published papers (Rozanski et al. 2016,
Nucleic Acids Res, DOI 10.1093/nar/gkv1148; Rozanski et al. 2019, Nucleic
Acids Res, DOI 10.1093/nar/gky1070):

1. NO PHENOTYPE CLASS EXISTS. There is no "Phenotype" or "RNAiPhenotype"
   class anywhere in the schema. "RNAi knockdown phenotype" data is
   really `RNAiResult.score`/`.description`/`.comment`/`.conditions` --
   a differential-expression measurement from RNA-seq under one specific
   published knockdown experiment (`RNAiExperiment`), not a curated,
   scored phenotype call. Coverage is limited to whatever genes happened
   to be assayed in the ~12 bulk RNA-seq-under-RNAi datasets PlanMine had
   indexed as of its 2019 release -- most genes will have NONE of this.
2. NO RELIABLE GENE SYMBOLS. PlanMine's own user guide states gene
   symbols are algorithmically inferred by orthology and should be
   treated only as an indicator of function, not a curated identity --
   the opposite of SubtiWiki's hand-curated gene names. This collector
   always keeps `primaryIdentifier` (the stable contig/gene ID) alongside
   any symbol, and never treats a missing or generic symbol as an error.
3. NO PATHWAY OR REGULON STRUCTURE OF ANY KIND. Nothing in this schema
   corresponds to SubtiWiki's Regulon/GeneRegulation classes. "Regulon/
   transcription-factor relationships" for S. mediterranea simply are not
   a queryable concept here -- do not expect this collector to return
   them, because the data does not exist upstream, not because of a
   parsing gap in this file.

Any downstream logic that scores confidence by "how complete is this
gene's record" needs to know PlanMine records are structurally *always*
going to look sparser than a SubtiWiki record, even for a well-studied
S. mediterranea gene -- that's a property of the source, not a signal
about the gene itself.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Optional

from acheron.collectors.base import BaseCollector
from acheron.models import Paper, PaperSource, SourceType

logger = logging.getLogger(__name__)

BASE_URL = "https://planmine.mpinat.mpg.de/planmine/service"
RESULTS_URL = f"{BASE_URL}/query/results"

ORGANISM = "S. mediterranea"

# Domain/GO-term substrings used to flag ion channel / innexin (gap
# junction) relevance in whatever real annotation text a gene actually
# has -- matched against real returned strings, never asserted otherwise.
ION_CHANNEL_KEYWORDS = ("innexin", "ion channel", "gap junction", "connexin", "channel")

MAX_RNAI_RESULTS_SHOWN = 10
MAX_DOMAIN_HITS_SHOWN = 10
MAX_HOMOLOGUES_SHOWN = 10


def _pathquery(view: list[str], constraint_path: str, op: str, value) -> str:
    """Build a minimal single-constraint InterMine PathQuery XML string.

    This is the standard InterMine PathQuery wire format (shared across
    every InterMine mine), kept deliberately simple -- one constraint per
    query -- specifically to avoid the multi-constraint constraintLogic
    syntax, which is harder to get exactly right without a live instance
    to test against.
    """
    query = ET.Element("query", {"name": "", "model": "genomic", "view": " ".join(view)})
    ET.SubElement(query, "constraint", {"path": constraint_path, "op": op, "value": str(value)})
    return ET.tostring(query, encoding="unicode")


class PlanMineCollector(BaseCollector):
    """Harvest S. mediterranea gene/homology/RNAi/domain records from PlanMine."""

    source_name = "planmine"

    def search(self, query: str, max_results: int = 50) -> list[Paper]:
        """Search PlanMine for genes matching `query` (a gene ID, or a
        topic word like "innexin" / "ion channel" matched against GO term
        names) and return one Paper per matching gene.
        """
        gene_ids = self._search_gene_ids(query, max_results)
        papers: list[Paper] = []
        for gene in gene_ids:
            paper = self._fetch_gene_paper(gene)
            if paper:
                papers.append(paper)

        logger.info("PlanMine: found %d gene records for '%s'", len(papers), query)
        return papers

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------
    def _run_query(self, view: list[str], constraint_path: str, op: str, value) -> list[list]:
        """Run one PathQuery and return its result rows (list of lists,
        each row's values in the same order as `view`). Returns an empty
        list on any failure -- collection continues with other queries
        rather than aborting the whole search.
        """
        xml = _pathquery(view, constraint_path, op, value)
        try:
            resp = self._get(RESULTS_URL, params={"query": xml, "format": "json"})
            data = resp.json()
        except Exception:
            logger.warning("PlanMine query failed (path=%s, op=%s, value=%s)", constraint_path, op, value)
            return []

        results = data.get("results")
        if results is None:
            logger.warning(
                "PlanMine response had no 'results' key -- response shape may differ "
                "from the standard InterMine format this collector assumes."
            )
            return []
        return results

    def _search_gene_ids(self, query: str, max_results: int) -> list[str]:
        gene_view = ["Gene.primaryIdentifier"]
        # Three separate simple lookups, merged -- see _pathquery's docstring
        # for why this avoids multi-constraint OR-logic XML.
        id_hits = self._run_query(gene_view, "Gene.primaryIdentifier", "CONTAINS", query)
        symbol_hits = self._run_query(gene_view, "Gene.symbol", "CONTAINS", query)
        go_hits = self._run_query(gene_view, "Gene.goAnnotation.ontologyTerm.name", "CONTAINS", query)

        ids: list[str] = []
        seen = set()
        for row in id_hits + symbol_hits + go_hits:
            if not row:
                continue
            gene_id = row[0]
            if gene_id is None or gene_id in seen:
                continue
            seen.add(gene_id)
            ids.append(gene_id)
            if len(ids) >= max_results:
                break
        return ids

    def _fetch_gene_paper(self, gene_id: str) -> Optional[Paper]:
        gene_view = [
            "Gene.primaryIdentifier",
            "Gene.secondaryIdentifier",
            "Gene.symbol",
            "Gene.briefDescription",
            "Gene.description",
            "Gene.organism.name",
            "Gene.organism.shortName",
            "Gene.organism.commonName",
        ]
        gene_rows = self._run_query(gene_view, "Gene.primaryIdentifier", "=", gene_id)
        if not gene_rows:
            return None
        gene_row = dict(zip(gene_view, gene_rows[0]))

        homology_view = ["Gene.homologues.type", "Gene.homologues.homologue.primaryIdentifier"]
        homology_rows = self._run_query(homology_view, "Gene.primaryIdentifier", "=", gene_id)

        rnai_view = [
            "Contig.rnaiResults.score",
            "Contig.rnaiResults.description",
            "Contig.rnaiResults.conditions",
            "Contig.rnaiResults.rnaiExperiment.name",
            "Contig.rnaiResults.rnaiExperiment.pubMedId",
        ]
        rnai_rows = self._run_query(rnai_view, "Contig.gene.primaryIdentifier", "=", gene_id)

        domain_view = [
            "Contig.domainHits.proteinDomain.description",
            "Contig.domainHits.proteinDomain.shortName",
        ]
        domain_rows = self._run_query(domain_view, "Contig.gene.primaryIdentifier", "=", gene_id)

        return self._parse_gene(gene_row, homology_rows, rnai_rows, domain_rows)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _parse_gene(
        self,
        gene: dict,
        homology_rows: list[list],
        rnai_rows: list[list],
        domain_rows: list[list],
    ) -> Optional[Paper]:
        gene_id = gene.get("Gene.primaryIdentifier")
        if not gene_id:
            return None

        symbol = gene.get("Gene.symbol") or ""
        organism_name = (
            gene.get("Gene.organism.name")
            or gene.get("Gene.organism.shortName")
            or ORGANISM
        )

        summary_lines = [f"Gene: {gene_id}"]
        if symbol:
            # PlanMine's own docs mark symbols as orthology-inferred, not
            # curated -- flag that here rather than presenting it as a
            # confirmed gene name the way SubtiWiki's names are.
            summary_lines.append(f"Symbol: {symbol} (PlanMine-inferred by orthology, not curated)")
        summary_lines.append(f"Organism: {organism_name}")

        if gene.get("Gene.briefDescription"):
            summary_lines.append(f"Brief description: {gene['Gene.briefDescription']}")
        if gene.get("Gene.description"):
            summary_lines.append(f"Description: {gene['Gene.description']}")

        # --- Homology annotations ---
        if homology_rows:
            shown = homology_rows[:MAX_HOMOLOGUES_SHOWN]
            homology_lines = []
            for row in shown:
                htype, target_id = (row + [None, None])[:2]
                if target_id:
                    line = f"{target_id}"
                    if htype:
                        line += f" ({htype})"
                    homology_lines.append(line)
            if homology_lines:
                summary_lines.append("Homology annotations: " + "; ".join(homology_lines))
                if len(homology_rows) > len(shown):
                    summary_lines.append(f"  ({len(homology_rows) - len(shown)} additional homologue(s) not shown)")

        # --- RNAi knockdown data (see module docstring: NOT a curated
        # phenotype call, just a differential-expression score from one
        # specific published RNA-seq-under-knockdown experiment) ---
        if rnai_rows:
            shown = rnai_rows[:MAX_RNAI_RESULTS_SHOWN]
            rnai_lines = []
            for row in shown:
                score, desc, conditions, exp_name, pubmed_id = (row + [None] * 5)[:5]
                parts = []
                if exp_name:
                    parts.append(exp_name)
                if score is not None:
                    parts.append(f"score={score}")
                if conditions:
                    parts.append(conditions)
                if desc:
                    parts.append(desc)
                line = ", ".join(str(p) for p in parts if p)
                if pubmed_id:
                    line += f" (PubMed {pubmed_id})"
                if line:
                    rnai_lines.append(line)
            if rnai_lines:
                summary_lines.append(
                    "RNAi/RNA-seq differential-expression results "
                    "(NOT a curated phenotype, see module docstring):"
                )
                summary_lines.extend(f"  - {line}" for line in rnai_lines)
                if len(rnai_rows) > len(shown):
                    summary_lines.append(f"  ({len(rnai_rows) - len(shown)} additional result(s) not shown)")

        # --- Protein domain hits, flagged for ion channel / innexin relevance ---
        ion_channel_hits = []
        if domain_rows:
            shown = domain_rows[:MAX_DOMAIN_HITS_SHOWN]
            domain_lines = []
            for row in shown:
                desc, short_name = (row + [None, None])[:2]
                label = desc or short_name
                if not label:
                    continue
                domain_lines.append(label)
                if any(kw in label.lower() for kw in ION_CHANNEL_KEYWORDS):
                    ion_channel_hits.append(label)
            if domain_lines:
                summary_lines.append("Protein domain annotations: " + "; ".join(domain_lines))
                if len(domain_rows) > len(shown):
                    summary_lines.append(f"  ({len(domain_rows) - len(shown)} additional domain hit(s) not shown)")

        if ion_channel_hits:
            summary_lines.append(
                "ION CHANNEL / GAP JUNCTION RELEVANT (matched domain annotation): "
                + "; ".join(ion_channel_hits)
            )

        abstract = "\n".join(summary_lines)

        return Paper(
            paper_id=f"planmine:{gene_id}",
            title=f"{gene_id} ({symbol}), PlanMine gene record" if symbol else f"{gene_id}, PlanMine gene record",
            authors=["PlanMine (Max Planck Institute)"],
            abstract=abstract,
            source=PaperSource.PLANMINE,
            journal="PlanMine",
            keywords=[gene_id] + ([symbol] if symbol else []),
            url=f"https://planmine.mpinat.mpg.de/planmine/report.do?id={gene_id}",
            organism=ORGANISM,
            source_type=SourceType.CURATED_DB,
        )
