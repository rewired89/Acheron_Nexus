"""Science-First evidence scoring and filtering for Nexus.

Implements:
  Stage C — Evidence Filtering with explicit scoring function.

Score = organism_match * 0.35
      + primary_data   * 0.25
      + measurement_specificity * 0.20
      + recency        * 0.10
      + citation_quality * 0.10

All weights are configurable but default to the canonical coefficients.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from acheron.models import QueryResult

logger = logging.getLogger(__name__)

# ======================================================================
# Organism taxonomy for phylogenetic-distance scoring
# ======================================================================
_ORGANISM_TIERS: dict[str, list[str]] = {
    "planarian": [
        "planari", "schmidtea", "dugesia", "girardia",
        "flatworm", "turbellari", "s. mediterranea",
        "d. japonica",
    ],
    "invertebrate": [
        "physarum", "hydra", "c. elegans", "caenorhabditis",
        "drosophila", "sea urchin", "echinoderm", "annelid",
        "axolotl", "regenerat",
    ],
    "xenopus": [
        "xenopus", "frog", "amphibian", "tadpole", "x. laevis",
    ],
    "vertebrate": [
        "mouse", "rat", "zebrafish", "mammal", "human", "organoid",
        "cell line", "hek293", "ipsc",
    ],
    "bacteria": [
        "bacillus", "b. subtilis", "subtilis", "bacteri", "prokaryot",
        "gram-positive", "gram-negative", "e. coli", "escherichia coli",
    ],
}


@dataclass
class EvidenceScore:
    """Breakdown of the evidence scoring function."""

    organism_match: float = 0.0
    primary_data: float = 0.0
    measurement_specificity: float = 0.0
    recency: float = 0.0
    citation_quality: float = 0.0
    total: float = 0.0
    reason: str = ""


@dataclass
class ScoredResult:
    """A QueryResult with its evidence score attached."""

    result: QueryResult
    score: EvidenceScore = field(default_factory=EvidenceScore)


# ======================================================================
# Scoring functions
# ======================================================================
def score_organism_match(
    text: str,
    target_organism: str = "planarian",
) -> float:
    """Score [0-1] how well the text matches the target organism.

    1.0 = exact organism match
    0.7 = close invertebrate / same regenerative class
    0.4 = Xenopus (common comparative model)
    0.2 = other vertebrate / organoid
    0.0 = no organism detected or irrelevant
    """
    lower = text.lower()

    # Exact match tier
    for keyword in _ORGANISM_TIERS.get(target_organism, []):
        if keyword in lower:
            return 1.0

    # Invertebrate tier
    for keyword in _ORGANISM_TIERS.get("invertebrate", []):
        if keyword in lower:
            return 0.7

    # Xenopus tier
    for keyword in _ORGANISM_TIERS.get("xenopus", []):
        if keyword in lower:
            return 0.4

    # Vertebrate tier
    for keyword in _ORGANISM_TIERS.get("vertebrate", []):
        if keyword in lower:
            return 0.2

    return 0.0


def score_primary_data(text: str) -> float:
    """Score [0-1] whether the source contains primary experimental data.

    1.0 = primary experimental data (methods, measurements, results)
    0.7 = has quantitative data but may be secondary
    0.4 = review / meta-analysis with cited data
    0.1 = opinion / commentary / theoretical
    """
    lower = text.lower()

    primary_signals = [
        "we measured", "we found", "we observed", "our results",
        "figure", "fig.", "table", "p <", "p=", "n =", "n=",
        "anova", "t-test", "wilcoxon", "mann-whitney",
        "standard deviation", "standard error", "s.e.m",
        "electrophysiol", "patch clamp", "whole-cell",
        "fluorescence imag", "confocal", "microscop",
    ]
    quant_signals = [
        "mv", "µm", "mm", "nm", "μa", "kohm", "mohm",
        "±", "mean", "median", "range",
    ]
    review_signals = [
        "review", "meta-analysis", "we summarize", "this review",
        "recent advances", "current understanding",
    ]

    primary_count = sum(1 for s in primary_signals if s in lower)
    quant_count = sum(1 for s in quant_signals if s in lower)
    review_count = sum(1 for s in review_signals if s in lower)

    if primary_count >= 3:
        return 1.0
    if quant_count >= 2 and primary_count >= 1:
        return 0.7
    if review_count >= 1:
        return 0.4
    if primary_count >= 1 or quant_count >= 1:
        return 0.5
    return 0.1


def score_measurement_specificity(text: str) -> float:
    """Score [0-1] whether the text contains specific quantitative measurements.

    Looks for: Vmem values, EF measurements, ion concentrations,
    gap junction conductance, specific gene expression quantification.
    """
    lower = text.lower()

    # Specific bioelectric measurements
    vmem_pattern = re.compile(r"-?\d+\.?\d*\s*m[vV]")
    ef_pattern = re.compile(r"\d+\.?\d*\s*(v/m|mv/mm|v/cm)")
    conc_pattern = re.compile(r"\d+\.?\d*\s*m[mM]")
    gj_pattern = re.compile(r"\d+\.?\d*\s*(ns|nS|pS|ps|kohm|mohm)")

    specificity_hits = 0
    if vmem_pattern.search(text):
        specificity_hits += 2  # Vmem is high-priority for Acheron
    if ef_pattern.search(text):
        specificity_hits += 2
    if conc_pattern.search(text):
        specificity_hits += 1
    if gj_pattern.search(text):
        specificity_hits += 2

    # Qualitative bioelectric mentions (less specific but relevant)
    qualitative_terms = [
        "depolariz", "hyperpolariz", "resting potential",
        "gap junction", "connexin", "innexin", "vmem",
        "membrane potential", "bioelectric",
    ]
    qual_count = sum(1 for t in qualitative_terms if t in lower)

    if specificity_hits >= 3:
        return 1.0
    if specificity_hits >= 1:
        return 0.7
    if qual_count >= 3:
        return 0.5
    if qual_count >= 1:
        return 0.3
    return 0.0


def score_recency(text: str) -> float:
    """Score [0-1] based on publication year heuristics.

    Recent publications score higher unless the source is a classic
    primary paper (detected by high citation signals).
    """
    # Try to extract year from metadata-style strings
    year_match = re.search(r"\b(19|20)\d{2}\b", text)
    if not year_match:
        return 0.3  # unknown recency, neutral score

    year = int(year_match.group())
    current_year = date.today().year

    age = current_year - year
    if age <= 2:
        return 1.0
    if age <= 5:
        return 0.8
    if age <= 10:
        return 0.5
    if age <= 20:
        return 0.3
    return 0.1


def score_citation_quality(result: QueryResult) -> float:
    """Score [0-1] based on citation metadata quality.

    Higher score for having DOI, PMID, PMCID, known authors.
    """
    quality = 0.0
    if result.doi:
        quality += 0.3
    if result.pmid:
        quality += 0.3
    if result.pmcid:
        quality += 0.2
    if result.authors and len(result.authors) > 0:
        quality += 0.1
    if result.paper_title and len(result.paper_title) > 10:
        quality += 0.1
    return min(quality, 1.0)


# ======================================================================
# Composite scoring
# ======================================================================
_WEIGHTS = {
    "organism_match": 0.35,
    "primary_data": 0.25,
    "measurement_specificity": 0.20,
    "recency": 0.10,
    "citation_quality": 0.10,
}


def score_evidence(
    result: QueryResult,
    target_organism: str = "planarian",
    weights: Optional[dict[str, float]] = None,
) -> EvidenceScore:
    """Compute the composite evidence score for a single QueryResult.

    score = organism_match*0.35 + primary_data*0.25
          + measurement_specificity*0.20 + recency*0.10
          + citation_quality*0.10
    """
    w = weights or _WEIGHTS
    full_text = f"{result.paper_title} {result.text}"

    om = score_organism_match(full_text, target_organism)
    pd = score_primary_data(result.text)
    ms = score_measurement_specificity(result.text)
    rc = score_recency(full_text)
    cq = score_citation_quality(result)

    total = (
        om * w.get("organism_match", 0.35)
        + pd * w.get("primary_data", 0.25)
        + ms * w.get("measurement_specificity", 0.20)
        + rc * w.get("recency", 0.10)
        + cq * w.get("citation_quality", 0.10)
    )

    return EvidenceScore(
        organism_match=round(om, 3),
        primary_data=round(pd, 3),
        measurement_specificity=round(ms, 3),
        recency=round(rc, 3),
        citation_quality=round(cq, 3),
        total=round(total, 3),
    )


def rank_and_filter(
    results: list[QueryResult],
    target_organism: str = "planarian",
    top_k: int = 8,
    min_score: float = 0.0,
    strict_organism: bool = True,
) -> list[ScoredResult]:
    """Score, filter, and rank retrieval results.

    Args:
        results: Raw QueryResult list from vectorstore.
        target_organism: Target organism for scoring.
        top_k: Number of results to return.
        min_score: Minimum composite score to include.
        strict_organism: If True, reject results with organism_match=0.

    Returns:
        Sorted list of ScoredResult (highest score first).
    """
    scored: list[ScoredResult] = []
    for r in results:
        s = score_evidence(r, target_organism)
        if s.total < min_score:
            continue
        if strict_organism and s.organism_match < 1.0:
            continue
        scored.append(ScoredResult(result=r, score=s))

    scored.sort(key=lambda x: x.score.total, reverse=True)
    return scored[:top_k]


def check_contradictions(
    results: list[ScoredResult],
) -> list[tuple[ScoredResult, ScoredResult, str]]:
    """Detect potential contradictions between sources.

    Returns list of (result_a, result_b, signal) where both mention
    opposing bioelectric effects (e.g., depolarization vs hyperpolarization
    for the same structure).
    """
    contradictions: list[tuple[ScoredResult, ScoredResult, str]] = []
    depol_pattern = re.compile(r"depolariz", re.IGNORECASE)
    hyperpol_pattern = re.compile(r"hyperpolariz", re.IGNORECASE)

    for i, a in enumerate(results):
        for b in results[i + 1:]:
            a_depol = bool(depol_pattern.search(a.result.text))
            a_hyper = bool(hyperpol_pattern.search(a.result.text))
            b_depol = bool(depol_pattern.search(b.result.text))
            b_hyper = bool(hyperpol_pattern.search(b.result.text))

            if (a_depol and b_hyper) or (a_hyper and b_depol):
                signal = (
                    "Polarity conflict: one source indicates depolarization, "
                    "the other hyperpolarization"
                )
                contradictions.append((a, b, signal))

    return contradictions
