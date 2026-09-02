"""Tests for RAG pipeline parsing and discovery logic."""

from acheron.rag.pipeline import (
    RAGPipeline,
    _try_parse_hypothesis,
    _try_parse_variable,
)


def test_parse_epistemic_sections():
    text = """
EVIDENCE
- Planarian regeneration involves bioelectric signals [1].
- Vmem gradients establish anterior-posterior polarity [2].

INFERENCE
- Gap junction connectivity likely encodes morphological targets [1][2].

SPECULATION
- Bioelectric patterns may serve as a computational substrate (medium confidence).

BIOELECTRIC SCHEMATIC
- Amputation -> Vmem depolarization -> Gj signaling -> Head regeneration

UNCERTAINTY
- No data on ion channel expression kinetics during first 6 hours.
"""
    _, evidence, inference, speculation, schematic = RAGPipeline._parse_epistemic_sections(text)
    assert len(evidence) == 2
    assert "bioelectric signals" in evidence[0]
    assert len(inference) == 1
    assert "Gap junction" in inference[0]
    assert len(speculation) == 1
    assert "computational substrate" in speculation[0]
    assert "Amputation" in schematic


def test_parse_epistemic_sections_markdown_headers():
    text = """
## EVIDENCE
- Fact one [1].

## INFERENCE
- Derived conclusion [1][2].

## SPECULATION
- Guess [low confidence].

## BIOELECTRIC SCHEMATIC
- Trigger -> Vmem change -> outcome
"""
    _, evidence, inference, speculation, schematic = RAGPipeline._parse_epistemic_sections(text)
    assert len(evidence) >= 1
    assert len(inference) >= 1
    assert len(speculation) >= 1
    assert "Trigger" in schematic


def test_parse_epistemic_sections_no_schematic():
    text = """
EVIDENCE
- Fact one [1].

INFERENCE
- Derived two [1][2].

SPECULATION
- Guess here.
"""
    _, evidence, inference, speculation, schematic = RAGPipeline._parse_epistemic_sections(text)
    assert len(evidence) >= 1
    assert schematic == ""


def test_parse_epistemic_sections_plain_summary():
    text = """
SIMPLE VERSION
- Gap junctions are like tiny doorways between neighboring cells.

EVIDENCE
- Gap junctions let ions pass directly between cells [1].
"""
    plain_summary, evidence, _, _, _ = RAGPipeline._parse_epistemic_sections(text)
    assert "doorways" in plain_summary
    assert len(evidence) == 1


def test_try_parse_variable_full():
    var = _try_parse_variable("Vmem=-45 (mV) [1]")
    assert var is not None
    assert var.name == "Vmem"
    assert var.value == "-45"
    assert var.unit == "mV"
    assert var.source_ref == "[1]"


def test_try_parse_variable_no_unit():
    var = _try_parse_variable("organism=planaria [2]")
    assert var is not None
    assert var.name == "organism"
    assert var.value == "planaria"
    assert var.unit == ""
    assert var.source_ref == "[2]"


def test_try_parse_variable_no_source():
    var = _try_parse_variable("conductance=12 (pS)")
    assert var is not None
    assert var.name == "conductance"
    assert var.value == "12"
    assert var.unit == "pS"


def test_try_parse_variable_not_a_variable():
    result = _try_parse_variable("This is just a sentence.")
    assert result is None


def test_try_parse_hypothesis_high_confidence():
    hyp = _try_parse_hypothesis(
        "Depolarization of wound-adjacent cells triggers regeneration (high confidence) [1][3]."
    )
    assert hyp is not None
    assert hyp.confidence == "high"
    assert "[1]" in hyp.supporting_refs
    assert "[3]" in hyp.supporting_refs


def test_try_parse_hypothesis_medium():
    hyp = _try_parse_hypothesis(
        "Gap junction blockers may impair anterior-posterior patterning (medium confidence)."
    )
    assert hyp is not None
    assert hyp.confidence == "medium"


def test_try_parse_hypothesis_default_low():
    hyp = _try_parse_hypothesis("Ion channels could store positional information.")
    assert hyp is not None
    assert hyp.confidence == "low"


def test_try_parse_hypothesis_too_short():
    result = _try_parse_hypothesis("Short.")
    assert result is None


def test_try_parse_hypothesis_prior_confidence():
    hyp = _try_parse_hypothesis(
        "Vmem depolarization drives blastema formation (Prior Confidence: high) [1][2]."
    )
    assert hyp is not None
    assert hyp.confidence == "high"
    assert "[1]" in hyp.supporting_refs


def test_try_parse_hypothesis_predicted_impact():
    hyp = _try_parse_hypothesis(
        "Gap junction blockade halts regeneration. "
        "Predicted Impact: confirms Gj as master patterning regulator [1]."
    )
    assert hyp is not None
    assert "Gj as master patterning regulator" in hyp.predicted_impact


def test_try_parse_hypothesis_assumptions():
    hyp = _try_parse_hypothesis(
        "Bioelectric signals encode positional memory. "
        "Assumptions: stable Vmem gradients exist; no confounding chemical signals"
    )
    assert hyp is not None
    assert len(hyp.assumptions) >= 1


def test_parse_discovery_output_new_sections():
    raw = """
1. EVIDENCE EXTRACTION
- Planarian head regeneration requires Vmem depolarization [1].
- Gap junctions propagate bioelectric signals [2].

2. VARIABLE EXTRACTION
- Vmem=-45 (mV) [1]
- Gj=0.5 (nS) [2]

3. PATTERN COMPARISON
- Both sources agree on bioelectric primacy in regeneration.

4. HYPOTHESES
- Vmem shift triggers blastema (high confidence) [1][2].

5. BIOELECTRIC SCHEMATIC
Amputation -> Vmem depolarization at wound -> Gj signal propagation -> Head regeneration

6. VALIDATION PATH
- Re-analyze voltage imaging datasets from Levin lab
- Computational Gj network simulation

7. CROSS-SPECIES NOTES
- Planarian innexins vs Xenopus connexins show conserved role in patterning

8. UNCERTAINTY
- No kinetics data for first 6 hours post-amputation
"""
    result = RAGPipeline._parse_discovery_output(
        raw_output=raw,
        query="planarian regeneration",
        sources=[],
        total_searched=50,
    )
    assert len(result.evidence) == 2
    assert len(result.variables) == 2
    assert len(result.inference) >= 1
    assert len(result.hypotheses) >= 1
    assert "Amputation" in result.bioelectric_schematic
    assert len(result.validation_path) == 2
    assert len(result.cross_species_notes) == 1
    assert len(result.uncertainty_notes) >= 1
