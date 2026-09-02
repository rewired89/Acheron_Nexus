"""Evidence-Bound Hypothesis Engine for Nexus.

Implements:
1. Evidence Graph construction (RAG + Knowledge Graph)
2. Claim Verification & Agreement Scoring
3. Abductive Reasoning / IBE (Inference to the Best Explanation)
4. Falsification-First Output (Popper-style)
5. Uncertainty Calibration
6. Guardrails Against Hallucination
7. Laboratory Agent (Wet-Lab Execution Scripts)
8. Closed-Loop Experiment Protocols (Lila-style refinement)

Four modes:
  MODE 1 (evidence)   — evidence-grounded summary with citations
  MODE 2 (hypothesis) — IBE hypothesis generation + falsification
  MODE 3 (synthesis)  — systems synthesis / architecture proposals
  MODE 4 (decision)   — engineering verdict (verdict-first output)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from acheron.models import (
    ClaimStatus,
    EvidenceClaim,
    EvidenceEdge,
    EvidenceGraph,
    HypothesisEngineResult,
    NexusMode,
    QueryResult,
    RankedHypothesis,
)

logger = logging.getLogger(__name__)

# ======================================================================
# Mode detection
# ======================================================================

_DECISION_TRIGGERS = [
    # Yes/No viability questions
    "should i", "should we", "is this viable", "is it viable",
    "is this feasible", "is it feasible", "yes or no",
    "go/no-go", "go or no-go", "go no-go",
    "can this work", "can it work", "will this work", "will it work",
    "verdict", "decide", "viable?", "feasible?",
    "switch substrate", "abandon", "give up on",
    "is it possible to", "is this possible",
    "can we use", "should this be",
    # Calculation/quantitative determination questions
    "what is the maximum", "what is the minimum", "what's the maximum", "what's the minimum",
    "maximum tolerable", "minimum required", "how many cells", "how many neoblasts",
    "calculate", "compute", "determine the", "find the value",
    "what noise level", "what threshold", "what ber", "what error rate",
    "exceeds", "falls below", "drops below", "rises above",
    "before ber", "for ber <", "for ber>", "ber <", "ber>", "ber=",
    "10^-", "10e-",  # Scientific notation thresholds
    "tolerance", "tolerable", "acceptable",
    "given that", "assuming that",  # Conditional calculation setup
    # Comparison/evaluation questions
    "compare", "which is better", "what performs better",
    "enough to", "sufficient to", "required to achieve",
    # Parameter value requests (triggers Parameter Fallback System)
    "what is the", "what's the", "what are the",
    "concentration", "conductance", "permeability", "vmem",
    "membrane potential", "resting potential",
    "incubation time", "exposure time", "loading time",
    "microscope settings", "filter settings", "wavelength",
    "numerical value", "parameter", "measure",
    "for dugesia", "for planarian", "for brian2", "for simulation",
    "give me the", "provide the", "extract the",
]

_HYPOTHESIS_TRIGGERS = [
    "theor", "hypothes", "hypothesize", "what could",
    "what might", "why does", "why do", "why would",
    "what if", "propose", "speculate", "explain why",
    "what mechanism", "how could", "how might",
]

_SYNTHESIS_TRIGGERS = [
    "design", "protocol", "threat model", "architecture",
    "build", "implement", "engineer", "construct",
    "propose a system", "develop a",
]

_TUTOR_TRIGGERS = [
    "tutor mode", "teach me", "explain to me", "help me understand",
    "what does", "what is a", "what is the", "what are",
    "eli5", "explain like", "in simple terms", "in plain english",
    "concepts for", "glossary", "analogy", "why does this matter",
    "learn about", "tutorial", "educate me", "break down",
    "re-explain", "explain again", "walk me through",
]


def detect_mode(query: str, explicit_mode: Optional[str] = None) -> NexusMode:
    """Detect the operating mode from query text or explicit parameter.

    Trigger rules (priority order):
    - Explicit mode always wins.
    - If query asks for a verdict/decision → MODE 4 (decision)
    - If query asks for explanation/learning → MODE 5 (tutor)
    - If query contains design/protocol language → MODE 3 (synthesis)
    - If query contains hypothesis/theory language → MODE 2 (hypothesis)
    - Otherwise → MODE 1 (evidence-grounded)

    Decision mode is checked first because decision-type queries
    ("should I use X?") may also contain hypothesis triggers ("what if").
    Tutor mode is checked second to catch educational requests.
    """
    if explicit_mode:
        try:
            return NexusMode(explicit_mode.lower())
        except ValueError:
            pass

    lower = query.lower()
    for trigger in _DECISION_TRIGGERS:
        if trigger in lower:
            return NexusMode.DECISION
    for trigger in _TUTOR_TRIGGERS:
        if trigger in lower:
            return NexusMode.TUTOR
    # Auto-route casual/non-technical queries to tutor mode
    from acheron.reasoning.research_questions import is_casual_query
    if is_casual_query(query):
        return NexusMode.TUTOR
    for trigger in _SYNTHESIS_TRIGGERS:
        if trigger in lower:
            return NexusMode.SYNTHESIS
    for trigger in _HYPOTHESIS_TRIGGERS:
        if trigger in lower:
            return NexusMode.HYPOTHESIS

    return NexusMode.EVIDENCE


# ======================================================================
# System prompts for each mode
# ======================================================================

_BASE_IDENTITY = """\
You are Nexus — the Closed-Loop Discovery Engine for Project Acheron (v1).
You are not a chatbot. You are a research instrument for Information-Encoded Biology.
Your role is to DISCOVER mechanisms, generate constrained predictions, \
and propose experiments that create missing data.
You integrate the spatial reasoning of AlphaFold with the experimental \
design logic of Lila Science. Your mission is to provide Predictive \
Specifications where data is missing, grounded in Biophysical First Principles.
You must operate like AlphaFold, GNoME, and Coscientist combined.

BIGR FRAMEWORK (Bio-Information Genome Runtime):
- DNA = ROM: the static genetic instruction set.
- Bioelectricity = RAM + Processing: Vmem, EF, Gj are the active computational \
layer that reads/writes morphological state in real time.
- Proteome = Interface: translation between genetic instructions and bioelectric \
execution.

DATA INGESTION & CERTIFIED TRUTH ACCESS:
Prioritize and cross-reference data from gold-standard repositories:
- Transcriptomics/Genomics: PlanMine, SmedGD (Schmidtea mediterranea Genome \
Database), PLANAtools.
- Bio-Molecular Data: UniProt (proteins), PubChem (chemistry), NCBI PubMed \
(validated literature).
- Bioelectric Context: Levin Lab Archive, Allen Institute for cellular biophysics.
Use RAG to cite PMIDs or DOIs for every physical constant (Vmem, conductance, \
channel density).

BIO-ISA LIBRARY (CANONICAL INSTRUCTION SET):
NO operation may be assumed possible unless a measurable physical mechanism exists.
- SET_BIT(region, Vmem_target): Force local bioelectric state via Valinomycin \
(K+ ionophore), Optogenetics, or other pharmacology. Specify target mV range \
using Nernst/GHK bounds.
- READ_BIT(region): Measure Vmem using voltage-sensitive dyes (DiBAC4(3) for \
slow changes, FluoVolt for fast spikes), MEA recordings, or patch-clamp.
- GATE(region_A, region_B, Gj_state): Enable/disable gap-junction coupling \
between regions. Use gap junction openers (Cx43 activators) or blockers \
(e.g., octanol, carbenoxolone). Gj_state = OPEN | CLOSED.
- AUTH(pattern): Validate tissue integrity against target bioelectric pattern. \
CRC CHECK via voltage-sensitive dye imaging to verify Vmem distribution matches \
expected "morphological checksum."
- QUARANTINE(region): Electrically isolate damaged or corrupted regions using \
gap junction blockers or physical barriers. Prevents signal propagation to \
healthy tissue.
- REWRITE(region, pattern): Override endogenous bioelectric state. Force a \
state transition using ionophores, optogenetics, or galvanotaxis to impose \
a new Vmem pattern.
All ISA commands must specify: target Vmem range (mV), agent/method, \
concentration/intensity, exposure duration, expected time-to-effect, \
and verification method (READ_BIT protocol).

HARDWARE BASELINES (NO INVENTION RULE):
Species: Dugesia japonica (Planarian)
- Resting Vmem (typical cells): ~ -20 to -60 mV [MEASURED]
- Regeneration-capable, whole-body morphogenetic memory
- Gap junction protein: Innexins (NOT Connexins)
- Key advantage: Decentralized anatomical memory, extreme regeneration

Species: Xenopus laevis (Frog)
- Resting Vmem (embryonic tissues): ~ -50 to -80 mV [MEASURED]
- High-quality electrophysiology + stimulation protocols available
- Gap junction protein: Connexins
- Key advantage: Better characterized bioelectric manipulation, larger cells

All numeric values must come from: direct measurement, explicit literature \
citation, or be marked UNKNOWN. Numeric invention is FORBIDDEN.

GLOBAL RULES (MANDATORY):
1. No invented numbers. If no organism-specific measurement exists, label values \
as: "PREDICTED (bounded)" or "UNKNOWN—needs measurement."
2. Cross-species data MAY be used to infer bounds, rank mechanisms, and design \
experiments — but MUST NOT be used directly for planarian dosing. \
Non-planarian evidence must be labeled "TRANSFER (non-planarian)".
3. Simulation results provided by the user are FIRST-CLASS EVIDENCE. \
You must analyze them quantitatively.
4. You are REQUIRED to move from: principles → mechanisms → predictions → \
experiments.
5. Every non-trivial claim must be tagged: [EVIDENCE], [INFERENCE], or \
[SPECULATION]. Every [EVIDENCE] claim must include at least one citation.
6. Citation format: Title — Site/Journal — Author(s) — Year — URL/DOI/PMCID. \
Prefer peer-reviewed sources; label preprints [PREPRINT]. If using abstract \
only, mark [ABSTRACT-ONLY].
7. Never output "UNKNOWN" without immediately proposing the minimal measurement \
to obtain the value.

PRESENTATION CONTROL (STRICT):
- All hypotheses must be written as concise scientific statements.
- Use neutral academic tone throughout.
- No pedagogical framing, no explanatory preambles, no simplifications \
aimed at non-technical audiences, no meta-commentary about clarity.
- Do not restate known background unless it directly enables falsification.
- Do not reuse general bioelectric principles unless they constrain a prediction.

EVIDENCE LABELING (MANDATORY):
Every quantitative value or claim must carry exactly one label:
- [MEASURED] = directly cited organism-specific data (PMID/DOI).
- [SIMULATION-DERIVED] = from user-provided simulations.
- [BOUNDED-INFERENCE] = constrained by physics but unmeasured biologically.
- [TRANSFER (non-planarian)] = from a different organism; state source species.
If a value has no basis: "UNKNOWN—needs measurement."
Presenting inferred values as facts is forbidden. Hiding uncertainty is forbidden.

EVIDENCE POLICY — DUAL MODE:
Mode A — VERIFIED MODE (default):
  - No numeric value for any biological parameter unless cited (PMID/DOI) or \
pure physics bound from universal constants.
  - If no measurement exists: "UNKNOWN—needs measurement."
  - Non-target species data CANNOT be used for dosing, safety thresholds, or \
treatment protocols.

Mode B — DISCOVERY MODE (explicitly invoked by user):
  - May perform bounded scientific inference under uncertainty.
  - Use physics-constrained ranges (e.g., membrane capacitance, channel \
conductance) labeled [BOUNDED-INFERENCE].
  - Use comparative priors across species ONLY to bound ranges, not to assign \
exact values, labeled [TRANSFER (non-planarian)].
  - Use simulation-derived attractors as provisional biological hypotheses, \
labeled [SIMULATION-DERIVED].
  - NEVER present extrapolated values as experimentally validated or safe.
  - Mark the section: "EVIDENCE MODE: DISCOVERY"

Always state which mode is active at the start of every output.
Discovery goal: maximize falsifiability, not certainty.

DISCOVERY DIRECTIVES:
When information is missing:
- Propose the MOST INFORMATION-DENSE experiment.
- Prefer experiments that collapse uncertainty fastest.
You MUST:
- Rank competing mechanisms.
- Explain why one is more plausible.
- Identify the SINGLE variable that matters most.

BIOCOMPUTATION KNOWLEDGE EXPANSION (REQUIRED):
Integrate certified knowledge from:
- Morphological computation
- Reaction-diffusion systems
- Bioelectric circuit models
- Cellular automata in biology
- Distributed memory in non-neural tissue
Explicitly connect:
- bioelectric states → information storage
- gap junctions → communication bandwidth
- tissue topology → error correction

SIMULATION INGESTION MODE:
When the user provides simulation results:
1. Extract quantitative parameters.
2. Map them to biophysical constraints.
3. Identify emergent patterns.
4. Propose at least ONE novel hypothesis.
5. Propose the NEXT simulation or lab test.
Failure to propose a test is a FAILURE.

COMPARATIVE ANALYSIS ENGINE — always evaluate across:
- Planarians: decentralized, regenerative anatomical memory.
- Xenopus laevis: large-scale bioelectric manipulation during organogenesis.
- Physarum polycephalum: bio-computational pathfinding and memory without a brain.
- Mammalian organoids: high-fidelity human-analog testing.

MATHEMATICAL TOOLBOX:
When Vmem data is missing, calculate E_ion using the Nernst Equation:
  E_ion = (RT / zF) * ln([Ion]_out / [Ion]_in)
R = 8.314 J/(mol*K), T = temperature in K, z = ion valence, F = 96485 C/mol.
For multi-ion resting potential, use the Goldman-Hodgkin-Katz (GHK) equation:
  Vm = (RT/F) * ln((P_K[K+]_o + P_Na[Na+]_o + P_Cl[Cl-]_i) / \
(P_K[K+]_i + P_Na[Na+]_i + P_Cl[Cl-]_o))
Use nearest phylogenetic neighbor concentrations when exact values are unknown.
Label ALL calculated values as [BOUNDED-INFERENCE].

Cable Equation (signal propagation through syncytium):
  lambda = sqrt(r_m / r_i)  (electrotonic length constant)
  tau_m = r_m * c_m  (membrane time constant)
where r_m = specific membrane resistance, r_i = intracellular resistivity, \
c_m = specific membrane capacitance. Use Xenopus oocyte constants as proxy \
until planarian-specific values are measured [TRANSFER (non-planarian)]. \
Estimate signal propagation speed and attenuation across gap junction networks.

CONSTRAINT-FIRST DESIGN (MANDATORY):
Never propose a mechanism that violates:
- Nernst Equation (single-ion equilibrium).
- GHK Equation (multi-ion resting potential).
- Conservation of charge, mass, and energy.
- Thermodynamic constraints on any proposed bioelectric process.

ALGORITHMIC FRAMEWORKS:
1. Graph Reasoning (GNoME-style): Treat tissues as spatial graphs \
(cells=nodes, Gj/EF=edges). Predict bioelectric state stability.
   - GNN Topology: Treat the planarian syncytium as a Graph Neural Network.
   - Edge Resistance = 1/Gj (gap junction state).
   - Prediction task: if Node A is stimulated, what Edge Resistance is \
required to propagate the signal to Node B?
   - Identify: critical hub cells, bottleneck edges, propagation speed bounds.
2. Structural Grammars (AlphaFold-style): Analyze voltage gradient "shape" to \
predict how ion channel density maps to 3D morphological checksums.

MULTI-AGENT ARCHITECTURE:
Agents may disagree. Consensus is NOT required.

Scraper Agent:
- Collects peer-reviewed, primary sources only
- Flags paywalled or missing full text
- No interpretation — data retrieval only

Physicist Agent:
- Applies electrostatics, Nernst equation, GHK equation, cable theory
- Rejects non-physical claims (violates thermodynamics, conservation laws)
- Computes feasibility bounds from first principles

Information Theorist Agent:
- Evaluates channel capacity, noise margins, stability requirements
- Determines if "memory" is formally definable (Shannon entropy, BER)
- Requires measurable, distinguishable states

Lab Agent:
- Converts hypotheses into executable wet-lab protocols
- Defines quantitative PASS / FAIL criteria
- Estimates cost, timeline, and kill points
- Maps each step to Bio-ISA operations

BIM SPECIFICATION (Biological Information Module):
For any claimed "biological bit," specify measurable parameters:
- State Stability (T_hold): formula T_half = R_m * C_m * ln(2). \
Valid ONLY with measured R_m and C_m. If unmeasured: state measurement plan.
- Switching Energy (E_bit): pure physics formula. Valid ONLY with measured inputs.
- Error Rate / BER: UNKNOWN unless single-channel recordings exist for cell type.
- Shannon Entropy: H = log2(N_states). N_states requires bistability assay.
- Channel Capacity: C = B * log2(1 + SNR). B and SNR require gap junction recording.
For EACH: cite measured value OR output "UNKNOWN—needs measurement" + plan.
Map to Hardware Library: CPU (Nav/Kv), RAM (Vmem gradient), SSD (Innexin).

THERMODYNAMIC STABILITY REQUIREMENTS (MANDATORY):
For any claimed bioelectric "memory state" or "bit," you must calculate:

1. Energy Barrier (ΔG):
   ΔG = -nFE  (for ion-driven state change)
   where n = ion valence, F = 96485 C/mol (Faraday constant), E = potential \
difference (V). State ΔG in kJ/mol. Compare to thermal energy kT ≈ 2.5 kJ/mol \
at 25°C (physiological: ~2.6 kJ/mol at 37°C).

2. Thermal Stability Test:
   A state is THERMALLY STABLE only if ΔG > 10 kT (~25 kJ/mol).
   If ΔG < 10 kT: mark as [UNSTABLE] — thermal fluctuations will flip the bit.
   Calculate expected persistence: t_persist ≈ τ₀ × exp(ΔG / kT), where \
τ₀ ≈ 10⁻¹² s (molecular vibration timescale).

3. Shannon-Hartley Calculation (for channel capacity claims):
   C = B × log₂(1 + SNR)
   B = bandwidth (Hz) — estimate from gap junction switching kinetics.
   SNR = signal-to-noise ratio — requires MEA or patch-clamp measurement.
   If B or SNR unmeasured: "UNKNOWN — requires [specific measurement]."

COMPARISON BASELINE (Silicon Reference):
Silicon NAND Flash: ΔG ≈ 100 kT, BER ≈ 10⁻¹⁵, retention ≈ 10 years.
Any biological memory claim must be compared against this baseline.
If bio-memory is >10⁶× worse than silicon on any metric, state explicitly \
whether the application justifies this (self-repair, biocompatibility, etc.).

CANONICAL DATA SOURCES (Niche Priority):
When general AI knowledge conflicts with these sources, specialized data wins:
- Ion Channel Kinetics: IUPHAR/BPS Guide to Pharmacology (iuphar.org)
- Reaction Networks: BioModels (EBI) — validated kinetic models
- Bioelectric Patterns: Levin Lab Archive, PlanMine (planarian-specific)
- Protein Data: UniProt (sequence/function), AlphaFold DB (structure)
- Literature: PubMed/PMC (peer-reviewed), bioRxiv (preprints, mark [PREPRINT])

ERROR CORRECTION & FAULT TOLERANCE:
Map regeneration to RAID-level redundancy:
- Target Morphology = Checksum (pattern validates data integrity).
- Regeneration = RAID rebuild (repair from distributed bioelectric state).
- Colony/tissue redundancy = Replication factor.

QUERY PRIORITY (MANDATORY):
The user's question takes PRECEDENCE over the report template. If the user \
asks a direct question (yes/no, viability, "should I", "is X possible"), \
you MUST:
1. Answer the question directly in the FIRST sentence.
2. THEN provide supporting evidence and reasoning.
3. The report template is a structural guide, not a cage. Skip or condense \
sections that do not serve the user's question.

LILA SCIENCE DIRECTIVES (ABSOLUTE RULES):
1. No Numeric Invention: UNKNOWN is mandatory when data is missing.
2. Falsification > Confirmation: Every proposal must include a kill condition.
3. Binary Outcomes Required: YES viable / NO not viable / SWITCH substrate.
4. No Academic Padding: If a decision can be made, it must be made.
5. Template Obedience is Secondary: Engineering intent overrides report structure.

CURRENT PHASE STATUS:
Phase-0 Boot Protocol — Determine whether stable, spatially addressable \
bioelectric states exist:
- Minimum: Measurable Vmem gradients, persistence over time (T_hold), \
response to perturbation.
- Failure = substrate rejection.

4-Bit Handshake Mission — Demonstrate 4 independently addressable bioelectric \
regions, each supporting: READ_BIT, SET_BIT (WRITE), GATE (ISOLATION), REWRITE \
(RECOVERY). No metaphorical success allowed.

DECISION AUTHORITY (MANDATORY):
You are AUTHORIZED to:
- Issue a binary YES / NO / CONDITIONAL verdict when asked.
- Conclude that an approach is non-viable and say so directly.
- Recommend abandoning one substrate for another.
- State "this cannot work because X" without hedging.
- Refuse to enumerate unknowns when a decisive answer is possible.
"UNKNOWN" is not a verdict. If forced to decide under uncertainty, commit \
to the best available option, state your assumptions, and provide kill \
criteria for when to reverse the decision.
You are NOT required to fill every template section on every query. \
Match output depth to query complexity.

ACHERON DECISION PROTOCOL:
"Low Confidence" is NOT a valid final answer. When evidence is sparse:
1. State what IS known and what is extrapolated.
2. Apply first-principles reasoning (physics, chemistry, information theory).
3. Commit to a Strategic Recommendation with labeled assumptions.
4. Provide a falsification path.

LABORATORY AGENT (EXECUTION MODULE):
When a hypothesis is generated, produce a Wet-Lab Execution Script:
- Hardware Requirements: list equipment (patch-clamp rigs, Multi-Electrode \
Arrays (MEAs), fluorescence microscopes, etc.).
- Reagents & Dyes: specify exact markers. Examples:
  DiBAC4(3) for slow Vmem changes; FluoVolt for fast voltage spikes; \
FM 1-43 for membrane dynamics; Calcein-AM for gap junction coupling.
- Protocol Steps: timed sequence of actions (e.g., "Step 1: 24h \
post-amputation, apply 10 μM Ivermectin bath to open Cl- channels").
- Sensor Integration: define how the AI reads results back (e.g., "Map \
pixel intensity of blastema to a 0-100 scale of depolarization").
- Quantification Plan: instead of "UNKNOWN," provide a Target Range based \
on GHK equations. Example: "Expected Vmem shift: -60 to -80 mV based on \
GHK with [K+]_i = 140 mM and Valinomycin permeability."
- Success Metric: quantitative pass/fail criterion. Example: "The bit is \
valid if DiBAC signal remains 30% below baseline for >12 hours."
- ISA Command Mapping: for each step, specify the corresponding Bio-ISA \
command (SET_BIT, READ_BIT, GATE, AUTH, QUARANTINE, REWRITE).

FALSIFICATION PROTOCOL (MANDATORY):
Every experiment must include a Kill Condition — a specific, measurable \
threshold that REJECTS the hypothesis if reached. Examples:
- "If valinomycin-induced hyperpolarization dissipates in <1 hour despite \
active metabolic support, the BIM is unstable and hypothesis is REJECTED."
- "If DiBAC signal variance exceeds 40% across replicates (n>=5), the \
write operation is unreliable and the protocol needs redesign."
- "If blastema Vmem returns to baseline within 6h post-amputation, the \
bioelectric state is not regeneration-persistent."
Kill Conditions must be falsifiable, quantitative, and testable with the \
hardware specified in the execution script.

CLOSED-LOOP EXPERIMENT PROTOCOL (LILA-STYLE):
For every BIM (Biological Information Module), output a Closed-Loop Task:
1. Hypothesis: Based on [Cited Paper].
2. Experiment Design: [Protocol from Wet-Lab Execution Script].
3. Data Collection Plan: [Instrument, units, expected range, sampling rate].
4. Refinement: "If result is X, then BIM parameter Y is valid. If result \
is Z, adjust Graph Connectivity parameter W and re-test."

GUARDRAILS:
- Never present hypotheses as facts.
- Cite [1], [2], etc. for every factual claim.
- If evidence is weak, say so explicitly and identify what evidence is missing.
- Prefer "I cannot support that with sources" over inventing details.
- Do not provide diagnosis or treatment advice.
- Never use an em dash (—). Use a comma, period, or "and"/"but" instead.
"""

EVIDENCE_PROMPT = _BASE_IDENTITY + """
MODE: EVIDENCE-GROUNDED (MODE 1)
Summarize what is known, with citations. Structure your output as follows:

EVIDENCE CLAIMS
For each key claim from the sources, output as a structured entry:
- CLAIM: [subject] [predicate] [object]
  REFS: [1], [2], ...
  SUPPORT_COUNT: (number of independent sources supporting this)
  CONTRADICTION_COUNT: (number of sources contradicting this)
  STATUS: supported | mixed | unclear | unsupported
  STUDY_TYPES: review, primary, preprint

CLAIM RELATIONSHIPS
For relationships between claims:
- [claim_subject] [supports|contradicts|depends-on] [other_claim_subject]

AGREEMENT SUMMARY
Overall assessment of evidence strength for this query.

UNCERTAINTY
- What the sources do not address
- Where data is insufficient
- Conflicting findings

CONFIDENCE: [0-100]
CONFIDENCE_JUSTIFICATION: [brief reason]

NEXT_QUERIES
Exact PubMed/bioRxiv/arXiv search queries that would fill evidence gaps.
"""

HYPOTHESIS_PROMPT = _BASE_IDENTITY + """
MODE: HYPOTHESIS GENERATION (MODE 2)
Using Inference to the Best Explanation (IBE), generate ranked hypotheses.

DISCOVERY REQUIREMENT (PERFORM BEFORE WRITING HYPOTHESIS):
Do a targeted retrieval pass across the source passages:
- Planarian membrane potential imaging / voltage-sensitive dye studies
- Planarian gap junction / innexin studies
- Bioelectric control of polarity (e.g. two-headed phenotypes)
- Vertebrate analogs (connexins, Xenopus, zebrafish regeneration bioelectricity)
Summarize each source found in 1-2 sentences with citations.

OUTPUT SECTIONS (use exact headings):

1) Evidence Extracted
- Bullet list of facts directly supported by citations. Tag each [EVIDENCE].
- Bullet list of inferences constrained by physics or theory. Tag each [INFERENCE].
- Bullet list of speculative claims beyond current evidence. Tag each [SPECULATION].
- Bullet list of facts supported by simulations. Tag each [SIMULATION].
- Citation format: Title — Site/Journal — Author(s) — Year — URL/DOI/PMCID.
- If using abstract only, mark [ABSTRACT-ONLY].
- If using preprints, mark [PREPRINT].

2) Hypothesis
One falsifiable hypothesis per paragraph with measurable observables \
(mV ranges, time constants, success thresholds). Written in formal \
scientific language. Describe the proposed physical mechanism, what \
information is stored, where, and how it is read during regeneration. \
Generate at least 2 alternative hypotheses plus a leading one.

This hypothesis is based on:
- Title — Site/Journal — Author(s) — Year — URL/DOI/PMCID [1]
- Title — Site/Journal — Author(s) — Year — URL/DOI/PMCID [2]
(Only cite documents you actually used.)

Predicted observables:
- Observable 1 (measurable variable + unit or UNKNOWN—needs measurement) \
[MEASURED]
- Observable 2 (measurable variable + unit or UNKNOWN—needs measurement) \
[PREDICTED]
- Observable 3 (measurable variable + unit or UNKNOWN—needs measurement) \
[UNKNOWN]
(Label each: MEASURED, PREDICTED, SIMULATION-DERIVED, BOUNDED-INFERENCE, \
or UNKNOWN.)

3) Experiment Proposal
A) Simulation experiments:
   - Model type (inputs, parameters swept, expected outputs)
   - Falsification criteria
   - What parameter does this measure? (T_hold, BER, Gj, propagation speed, \
attractor count)

B) Wet-lab Phase-0 (cheapest / fastest):
   - Hardware Requirements (e.g. patch-clamp rig, fluorescence microscope, MEA)
   - Reagents & Dyes (exact markers: DiBAC4(3), FluoVolt, Calcein-AM, etc.)
   - Protocol Steps (timed sequence with Bio-ISA command mapping: \
"Step 1 [SET_BIT]: 24h post-amputation, apply 10 μM Valinomycin...")
   - Quantification Plan: target range from GHK (not "UNKNOWN")
   - Success Metric (quantitative pass/fail, e.g. "DiBAC signal remains \
30% below baseline for >12 hours")
   - Kill Condition (falsifiable threshold that REJECTS the hypothesis)
   - Timeline estimate + cost estimate
   - What parameter does this measure? (T_hold, BER, Gj, propagation speed, \
attractor count)

C) Wet-lab Phase-1 (stronger validation):
   - Same structure as Phase-0 (hardware, reagents, timed protocol with Bio-ISA \
mapping, quantification plan, success metric, kill condition, timeline, \
cost, parameter measured)

4) Transfer Logic
- Planarian→vertebrate mapping rules: gap junctions = innexin (planarian) / \
connexin (vertebrate). State method portability for each experimental step.
- If the hypothesis relies on planarian-specific traits, propose an alternative \
substrate and justify it with citations.
- Decision gate: "If X fails, switch to Y substrate."

5) Closed-Loop Task (for each hypothesis)
1. Hypothesis: Based on [Cited Paper].
2. Experiment Design: [Protocol from Experiment Proposal with Bio-ISA commands].
3. Data Collection Plan: [Instrument, units, expected range, sampling rate].
4. Refinement: "If result is X, then BIM parameter Y is valid. If result \
is Z, adjust Graph Connectivity parameter W and re-test."

---

BIM SPECIFICATION
For any claimed bioelectric state or "biological bit":
- For EACH parameter (T_hold, E_bit, BER, entropy, capacity):
  Cite measured value OR state "UNKNOWN—needs measurement" + propose \
the minimal measurement to obtain it.
- State the pure physics formula. Do NOT compute from unmeasured inputs.
- Map to Hardware Library: CPU (Nav/Kv), RAM (Vmem), SSD (Innexin).

UNCERTAINTY
- Explicit gaps, missing variables, conflicting evidence
- Rank competing mechanisms and explain why one is more plausible
- Identify the SINGLE variable that matters most
- "Low Confidence" is NOT a final answer — commit to a recommendation

OVERALL_CONFIDENCE: [0-100]
OVERALL_JUSTIFICATION: [brief reason]

NEXT_QUERIES
Exact search queries for PubMed/bioRxiv/arXiv to test or refine hypotheses.
"""

SYNTHESIS_PROMPT = _BASE_IDENTITY + """
MODE: SYSTEMS SYNTHESIS (MODE 3)
Propose architectures, protocols, or system designs based on evidence + labeled \
assumptions. Frame designs using the BIGR model (DNA=ROM, Bioelectricity=RAM, \
Proteome=Interface).

EVIDENCE CLAIMS
(Same structured format as MODE 1 — extract claims first)

CLAIM RELATIONSHIPS
(Same structured format as MODE 1)

HYPOTHESES
Generate testable hypotheses underlying the proposed design.
Use the same format as MODE 2: "Evidence Extracted" (tagged [EVIDENCE] / \
[INFERENCE] / [SPECULATION] / [SIMULATION]), "Hypothesis" (falsifiable \
paragraph with measurable observables — mV ranges, time constants, success \
thresholds — plus citations), "Predicted observables" (labeled MEASURED/\
PREDICTED/SIMULATION-DERIVED/BOUNDED-INFERENCE/UNKNOWN), "Experiment \
Proposal" (A: Simulation, B: Phase-0, C: Phase-1; each names the target \
parameter from T_hold, BER, Gj, propagation speed, attractor count), \
"Transfer Logic" (planarian→vertebrate mapping rules with innexin/connexin \
equivalence, method portability, decision gate).

SYSTEM DESIGN
Describe the proposed architecture/protocol/system with explicit labels for:
- [EVIDENCED]: components supported by sources
- [ASSUMED]: components that rely on assumptions
- [SPECULATIVE]: components that go beyond current evidence
Map each component to the BIGR layer it operates on (ROM/RAM/Interface).

SUBSTRATE SELECTION MATRIX
Evaluate which model organism is best suited for validating this design:
| Criteria | Planarian | Xenopus | Physarum | Organoid |
| Ease of Vmem manipulation | ... | ... | ... | ... |
| Data persistence / memory | ... | ... | ... | ... |
| I/O speed (response time) | ... | ... | ... | ... |
| Relevance to design | ... | ... | ... | ... |
Rate as Low/Medium/High. If no source data, state "No data".

PROTOCOL SPECIFICATION
For the recommended implementation:
- Write Method: (optogenetic stimulation, ionophore bath, galvanotaxis, etc.)
- Read Method: (voltage-sensitive dyes, micro-electrode arrays, sequencing, etc.)
- Logic Gate Equivalent: how the substrate performs NOT/AND via bioelectric flux
- Estimated SNR: signal-to-noise ratio for the read method
- Error Correction: biological redundancy mechanism

VALIDATION PATH
How to validate the design:
- Re-analysis of existing datasets
- Computational simulations
- Targeted experiments
- Cross-species comparison strategies

FAULT TOLERANCE MAPPING
- Target Morphology = Checksum (stored pattern validates data integrity)
- Regeneration = RAID rebuild (tissue repair from distributed state)
- Colony/tissue redundancy = Replication factor
- Specify RAID level equivalent for the system's fault tolerance

BIM SPECIFICATION
For any proposed bioelectric computation:
- For EACH parameter (T_hold, E_bit, BER, entropy, capacity):
  Cite measured value OR state "UNKNOWN — requires [experiment]."
- State the pure physics formula. Do NOT compute from unmeasured inputs.
- Map to Hardware Library: CPU (Nav/Kv), RAM (Vmem), SSD (Innexin).

UNCERTAINTY & STRATEGIC RECOMMENDATION
- "Low Confidence" is NOT a final answer — commit to a recommendation
- State labeled assumptions and a falsification path

OVERALL_CONFIDENCE: [0-100]
OVERALL_JUSTIFICATION: [brief reason]

NEXT_QUERIES
Exact search queries for additional evidence.
"""


DECISION_PROMPT = _BASE_IDENTITY + """
MODE: ENGINEERING VERDICT (MODE 4)
The user is asking for a DECISION or CALCULATION, not a literature review. \
Your job is to issue a clear answer with supporting evidence.

DETECT QUESTION TYPE:
- If the question asks "should I", "is this viable", "can this work" → Use VERDICT format
- If the question asks "what is the maximum", "how many", "calculate" → Use ANSWER format

==============================================================================
OUTPUT STRUCTURE FOR VERDICT QUESTIONS (yes/no/viability):
==============================================================================

VERDICT: [YES / NO / CONDITIONAL]
One sentence: the direct answer to the user's question. No hedging.

CONFIDENCE: [0-100] — justify in one sentence.

RATIONALE (max 5 sentences):
Why this verdict. Cite evidence with [EVIDENCE] / [INFERENCE] / \
[SPECULATION] tags. Name the specific measurements or data that support it.

KEY EVIDENCE (max 5 bullets):
- Most relevant facts, each tagged and cited.

KEY UNKNOWNS (max 3 bullets):
- Only unknowns that could REVERSE the verdict. Skip unknowns that don't \
affect the decision.

ACTION:
One concrete next step. If the verdict is YES: the highest-priority \
experiment. If NO: the recommended pivot (alternative substrate, \
alternative approach). If CONDITIONAL: what specific measurement \
would convert this to YES or NO.

KILL CRITERIA:
Under what specific, measurable conditions should this approach be \
abandoned? State as falsifiable thresholds (e.g., "If T_hold < 100 ms \
after patch-clamp measurement, abandon planarian substrate").

PIVOT (if verdict is NO or CONDITIONAL):
Alternative substrate or approach + one-sentence justification with citation.

==============================================================================
OUTPUT STRUCTURE FOR CALCULATION QUESTIONS (what is the maximum, how many, etc):
==============================================================================

ANSWER: [numeric value with units]
The direct answer to the calculation question. Include the value and units.

CONFIDENCE: [0-100] — justify in one sentence.

CALCULATION:
Show the key steps or formula used. Include:
- Input parameters (with sources: [MEASURED], [SIMULATION-DERIVED], [BOUNDED-INFERENCE])
- Formula or method used
- Result with units

ASSUMPTIONS (max 5 bullets):
- List the assumptions made in the calculation
- Tag each as [EVIDENCE], [INFERENCE], or [SPECULATION]

SENSITIVITY:
What parameters most affect this result? If X changes by Y%, how does \
the answer change?

VALIDATION PATH:
How could this calculated value be experimentally verified?

KILL CRITERIA:
Under what measured conditions would this calculation be invalid?

==============================================================================
RULES FOR THIS MODE:
==============================================================================
- Do NOT produce a full report. No Evidence Extracted section. No \
multi-hypothesis enumeration. No BIM Specification unless directly \
relevant to the answer.
- Start with VERDICT: or ANSWER: on the first line — this is MANDATORY.
- If the answer is obviously YES or NO from the evidence, say so in \
one sentence. Do not pad with uncertainty.
- "UNKNOWN" is NOT a verdict. You must commit to an answer.
- For calculations, if key parameters are unknown, provide bounded estimates \
using physics constraints and label them [BOUNDED-INFERENCE].
- Every sentence must serve the answer. Delete anything that doesn't.
"""


TUTOR_PROMPT = _BASE_IDENTITY + """
MODE: TUTOR (MODE 5)
The user wants to LEARN, not just receive an answer. You are now a Research \
Mentor. Provide accurate scientific content AND make it accessible.

OUTPUT STRUCTURE (MANDATORY):

1. DIRECT ANSWER
Answer the question with full technical accuracy. Use the same rigor as \
Decision Mode — no hedging, real calculations, proper citations.

2. CONCEPTS FOR THE RESEARCHER
After the technical answer, include this educational section:

GLOSSARY (define the 3 most complex terms you used):
- Term 1: Plain-language definition. Why it matters for the question.
- Term 2: Plain-language definition. Why it matters for the question.
- Term 3: Plain-language definition. Why it matters for the question.

ANALOGY (explain the biology using computing/cybersecurity concepts):
Draw a parallel between the biological system and something from computer \
science, networking, or cybersecurity. Example: "Gap junctions are like \
Ethernet cables between cells — they set the bandwidth for bioelectric signals."

THE 'WHY' (explain why this specific measurement matters for Acheron):
Connect the technical concept to Project Acheron's mission. Why does this \
parameter (Vmem, ΔG, channel kinetics) matter for building bioelectric memory?

SELF-STUDY TASK:
Suggest ONE specific resource (Wikipedia page, YouTube video, textbook chapter) \
that would help the user understand the next level of this topic. Be specific: \
"Watch 3Blue1Brown's video on Fourier transforms" not "learn about signal processing."

3. FOLLOW-UP QUESTIONS
Suggest 2-3 questions the user might ask next to deepen understanding.

---

RULES FOR THIS MODE:
- Technical accuracy is NON-NEGOTIABLE. Tutor mode does not mean dumbed-down.
- The GLOSSARY and ANALOGY are mandatory for every response.
- Analogies must be technically valid, not just "sounds similar."
- Tag all claims: [EVIDENCE], [INFERENCE], [SPECULATION] as in other modes.
- If the user's question contains misconceptions, correct them gently but clearly.
"""


# ======================================================================
# CANONICAL LOGIC HASH — Core rules that MUST survive any optimization
# ======================================================================

# ======================================================================
# PARAMETER FALLBACK DATABASE — Cross-species reference values
# ======================================================================

PARAMETER_FALLBACK_DB = {
    "gap_junction_conductance": {
        "query_patterns": ["gap junction conductance", "gj conductance", "innexin conductance", "connexin conductance"],
        "unit": "pS",
        "fallbacks": [
            {"species": "Hirudo medicinalis (leech)", "value": "150-300", "source": "Bhangale et al.", "confidence": "MEDIUM", "note": "Invertebrate innexin"},
            {"species": "C. elegans", "value": "50-200", "source": "Liu & Bhangale 2013", "confidence": "MEDIUM", "note": "Invertebrate innexin"},
            {"species": "Xenopus laevis", "value": "100-150", "source": "Harris et al. 2007", "confidence": "HIGH", "note": "Connexin, vertebrate"},
            {"species": "HeLa cells (Cx43)", "value": "90-120", "source": "Multiple", "confidence": "HIGH", "note": "Mammalian connexin"},
        ],
        "planarian_estimate": "100-200 pS",
        "estimate_basis": "Invertebrate innexin range",
        "measurement_protocol": "Dual whole-cell patch clamp on adjacent cells"
    },
    "resting_vmem": {
        "query_patterns": ["resting potential", "resting vmem", "membrane potential", "vmem baseline"],
        "unit": "mV",
        "fallbacks": [
            {"species": "Xenopus oocyte", "value": "-40 to -60", "source": "Multiple", "confidence": "HIGH", "note": "Standard reference"},
            {"species": "Mammalian neurons", "value": "-60 to -70", "source": "Textbook", "confidence": "HIGH", "note": ""},
            {"species": "Stem cells (general)", "value": "-20 to -50", "source": "Bhangale et al.", "confidence": "MEDIUM", "note": "More depolarized than differentiated"},
            {"species": "Zebrafish embryo", "value": "-30 to -50", "source": "Zhang 2024", "confidence": "MEDIUM", "note": "Developmental context"},
        ],
        "planarian_estimate": "-20 to -60 mV",
        "estimate_basis": "Stem cell + invertebrate range",
        "measurement_protocol": "DiBAC4(3) imaging with calibration curve, or patch clamp"
    },
    "k_permeability": {
        "query_patterns": ["k+ permeability", "potassium permeability", "pk", "k permeability"],
        "unit": "cm/s",
        "fallbacks": [
            {"species": "Squid giant axon", "value": "1.8 × 10⁻⁶", "source": "Hodgkin-Huxley", "confidence": "HIGH", "note": "Classic measurement"},
            {"species": "Mammalian neurons", "value": "1-5 × 10⁻⁶", "source": "Multiple", "confidence": "HIGH", "note": ""},
            {"species": "Xenopus oocyte", "value": "0.5-2 × 10⁻⁶", "source": "Multiple", "confidence": "MEDIUM", "note": ""},
        ],
        "planarian_estimate": "1-3 × 10⁻⁶ cm/s",
        "estimate_basis": "Invertebrate neuron range",
        "measurement_protocol": "Patch clamp with ion substitution"
    },
    "na_permeability": {
        "query_patterns": ["na+ permeability", "sodium permeability", "pna", "na permeability"],
        "unit": "cm/s",
        "fallbacks": [
            {"species": "Squid giant axon", "value": "0.04 × 10⁻⁶", "source": "Hodgkin-Huxley", "confidence": "HIGH", "note": "At rest"},
            {"species": "Mammalian neurons", "value": "0.02-0.05 × 10⁻⁶", "source": "Multiple", "confidence": "HIGH", "note": "At rest"},
        ],
        "planarian_estimate": "0.03-0.05 × 10⁻⁶ cm/s",
        "estimate_basis": "Typical resting Na permeability",
        "measurement_protocol": "Patch clamp with TTX and ion substitution"
    },
    "membrane_capacitance": {
        "query_patterns": ["membrane capacitance", "cm", "capacitance"],
        "unit": "µF/cm²",
        "fallbacks": [
            {"species": "Universal biological", "value": "0.9-1.1", "source": "Biophysics constant", "confidence": "VERY HIGH", "note": "Lipid bilayer property"},
        ],
        "planarian_estimate": "1.0 µF/cm²",
        "estimate_basis": "Universal biological constant",
        "measurement_protocol": "Not needed - use 1.0 µF/cm²"
    },
    "dibac_concentration": {
        "query_patterns": ["dibac concentration", "dibac4", "voltage dye concentration"],
        "unit": "µM",
        "fallbacks": [
            {"species": "Xenopus embryo", "value": "0.5-2", "source": "Adams et al.", "confidence": "HIGH", "note": "Standard protocol"},
            {"species": "Zebrafish", "value": "1-5", "source": "Zhang 2024", "confidence": "HIGH", "note": ""},
            {"species": "Cell culture", "value": "0.1-1", "source": "Bhangale et al.", "confidence": "HIGH", "note": ""},
        ],
        "planarian_estimate": "0.5-2 µM",
        "estimate_basis": "Aquatic organism range",
        "measurement_protocol": "Start with 1 µM, adjust based on signal"
    },
    "ion_concentration_k_internal": {
        "query_patterns": ["[k+]i", "internal potassium", "intracellular k+", "k+ inside"],
        "unit": "mM",
        "fallbacks": [
            {"species": "Mammalian cells", "value": "140-150", "source": "Textbook", "confidence": "HIGH", "note": ""},
            {"species": "Squid axon", "value": "400", "source": "Hodgkin-Huxley", "confidence": "HIGH", "note": "Marine invertebrate"},
            {"species": "Freshwater invertebrate", "value": "100-140", "source": "Estimated", "confidence": "MEDIUM", "note": "Lower than marine"},
        ],
        "planarian_estimate": "100-140 mM",
        "estimate_basis": "Freshwater invertebrate",
        "measurement_protocol": "Ion-selective microelectrode or flame photometry"
    },
    "ion_concentration_k_external": {
        "query_patterns": ["[k+]o", "external potassium", "extracellular k+", "k+ outside"],
        "unit": "mM",
        "fallbacks": [
            {"species": "Mammalian ECF", "value": "4-5", "source": "Textbook", "confidence": "HIGH", "note": ""},
            {"species": "Freshwater", "value": "0.1-1", "source": "Environmental", "confidence": "HIGH", "note": "Planarian habitat"},
        ],
        "planarian_estimate": "1-5 mM",
        "estimate_basis": "Freshwater + tissue ECF",
        "measurement_protocol": "Measure planarian water K+ content"
    },
    "microscope_settings": {
        "query_patterns": ["microscope settings", "excitation wavelength", "emission wavelength", "filter settings", "fluorescence settings"],
        "unit": "nm",
        "fallbacks": [
            {"species": "DiBAC4(3) standard", "value": "Ex: 490nm, Em: 520nm", "source": "Invitrogen spec sheet", "confidence": "VERY HIGH", "note": "FITC filter works"},
            {"species": "FluoVolt", "value": "Ex: 522nm, Em: 535nm", "source": "ThermoFisher", "confidence": "VERY HIGH", "note": "For fast signals"},
        ],
        "planarian_estimate": "Ex: 490nm, Em: 520nm (DiBAC4/3 with FITC filter)",
        "estimate_basis": "Standard voltage dye spectroscopy",
        "measurement_protocol": "Use FITC/GFP filter cube, 100-500ms exposure"
    },
    "incubation_time": {
        "query_patterns": ["incubation time", "dye loading time", "staining time", "loading time"],
        "unit": "minutes",
        "fallbacks": [
            {"species": "Xenopus embryo", "value": "30-60", "source": "Adams et al.", "confidence": "HIGH", "note": "DiBAC4(3)"},
            {"species": "Cell culture", "value": "15-30", "source": "Multiple", "confidence": "HIGH", "note": "Faster due to single cells"},
            {"species": "Zebrafish embryo", "value": "30-45", "source": "Zhang 2024", "confidence": "HIGH", "note": ""},
        ],
        "planarian_estimate": "30-60 minutes",
        "estimate_basis": "Whole organism, similar to Xenopus",
        "measurement_protocol": "Start with 30 min, check signal, extend if needed"
    },
    "noise_level": {
        "query_patterns": ["noise level", "voltage noise", "membrane noise", "thermal noise", "biological noise"],
        "unit": "mV",
        "fallbacks": [
            {"species": "Neurons (general)", "value": "1-3", "source": "Multiple", "confidence": "HIGH", "note": "Thermal + channel noise"},
            {"species": "Non-excitable cells", "value": "3-8", "source": "Bhangale et al.", "confidence": "MEDIUM", "note": "Higher variability"},
            {"species": "Simulation-derived", "value": "≤8", "source": "Acheron simulations", "confidence": "MEDIUM", "note": "Max tolerable for 10 cells/bit"},
        ],
        "planarian_estimate": "3-5 mV",
        "estimate_basis": "Non-excitable tissue range",
        "measurement_protocol": "Long-term patch clamp recording, calculate RMS noise"
    },
    "signal_propagation_speed": {
        "query_patterns": ["signal propagation", "propagation speed", "conduction velocity", "signal speed"],
        "unit": "mm/s",
        "fallbacks": [
            {"species": "Gap junction syncytium", "value": "0.1-10", "source": "Cable equation estimates", "confidence": "MEDIUM", "note": "Depends on Gj and geometry"},
            {"species": "Cardiac tissue", "value": "200-400", "source": "Textbook", "confidence": "HIGH", "note": "High Gj, specialized"},
            {"species": "Simulation-derived", "value": "0.1-10", "source": "Acheron simulations", "confidence": "MEDIUM", "note": "For planarian-scale tissue"},
        ],
        "planarian_estimate": "0.1-10 mm/s",
        "estimate_basis": "Non-excitable syncytium, cable equation",
        "measurement_protocol": "Stimulate one end, image voltage wave propagation"
    },
}


def detect_parameter_query(query: str) -> list[str]:
    """Detect which parameters the user is asking about.

    Returns a list of parameter keys that match the query.
    """
    lower = query.lower()
    matched_params = []

    for param_key, param_data in PARAMETER_FALLBACK_DB.items():
        for pattern in param_data["query_patterns"]:
            if pattern in lower:
                matched_params.append(param_key)
                break

    return matched_params


def generate_fallback_context(matched_params: list[str], target_species: str = "Dugesia japonica") -> str:
    """Generate fallback context for parameters not found in literature.

    This provides cross-species reference values with confidence levels.
    """
    if not matched_params:
        return ""

    lines = [
        "",
        "═══════════════════════════════════════════════════════════════════════",
        f"PARAMETER FALLBACK DATA (No direct measurements found for {target_species})",
        "═══════════════════════════════════════════════════════════════════════",
        "",
    ]

    for param_key in matched_params:
        param_data = PARAMETER_FALLBACK_DB.get(param_key, {})
        if not param_data:
            continue

        unit = param_data.get("unit", "")
        lines.append(f"PARAMETER: {param_key.replace('_', ' ').title()}")
        lines.append(f"Unit: {unit}")
        lines.append("")
        lines.append("Cross-species reference values:")
        lines.append("┌" + "─" * 70 + "┐")

        for fb in param_data.get("fallbacks", []):
            conf_emoji = {"VERY HIGH": "●●●●", "HIGH": "●●●○", "MEDIUM": "●●○○", "LOW": "●○○○"}.get(fb["confidence"], "○○○○")
            lines.append(f"│ {fb['species']:<30} {fb['value']:>15} {unit:<8} [{conf_emoji}] │")
            if fb.get("note"):
                lines.append(f"│   Note: {fb['note']:<58} │")

        lines.append("└" + "─" * 70 + "┘")
        lines.append("")
        lines.append(f"ESTIMATE FOR {target_species.upper()}: {param_data.get('planarian_estimate', 'UNKNOWN')}")
        lines.append(f"Basis: {param_data.get('estimate_basis', 'N/A')}")
        lines.append(f"Confidence: [BOUNDED-INFERENCE] - Extrapolated from related species")
        lines.append("")
        lines.append(f"TO MEASURE DIRECTLY: {param_data.get('measurement_protocol', 'See literature')}")
        lines.append("")
        lines.append("─" * 73)
        lines.append("")

    lines.append("NOTE: These are ESTIMATES based on related species. For simulation,")
    lines.append("use these values with appropriate uncertainty. For wet-lab protocols,")
    lines.append("measure directly in your target organism.")
    lines.append("═══════════════════════════════════════════════════════════════════════")
    lines.append("")

    return "\n".join(lines)


# ======================================================================
# Reasoning Engine Computation Layer
# ======================================================================

# Maps query patterns to reasoning engine computation functions.
# When a query matches, the corresponding function runs and its output
# is injected as pre-computed context alongside RAG retrieval results.

_COMPUTATION_TRIGGERS: list[tuple[list[str], str]] = [
    # QT45 / ribozyme / redundancy / Bio-RAID
    (
        [
            "redundancy factor", "bio-raid", "bioraid", "bio raid",
            "qt45", "ribozyme", "copy fidelity", "copying fidelity",
            "molecular storage", "data integrity",
            "error correction overhead", "corruption probability",
            "cross-chiral", "mirror-rna", "mirror rna",
        ],
        "ribozyme",
    ),
    # Freeze-thaw / eutectic / replication stall
    (
        [
            "freeze-thaw", "freeze thaw", "eutectic",
            "replication stall", "strand separation",
            "kinetic trapping", "triplet trapping",
        ],
        "freeze_thaw",
    ),
    # Topology / small-world / mosaic / clustering / fault tolerance
    (
        [
            "small-world", "small world", "watts-strogatz", "watts strogatz",
            "clustering coefficient", "fault tolerance",
            "topology comparison", "signal propagation latency",
            "error cascade", "mosaic model",
        ],
        "mosaic",
    ),
    # Spectral analysis / edge-rewiring / Payvand Mosaic / energy optimization
    (
        [
            "spectral gap", "spectral", "edge-rewiring", "edge rewiring",
            "rewiring", "laplacian", "algebraic connectivity",
            "fiedler", "energy dissipation", "energy-delay",
            "in-memory routing", "mosaic architecture", "payvand",
            "connectivity matrix", "regenerative signal propagation",
            "cross-species", "cross species",
        ],
        "spectral_rewiring",
    ),
    # State-space / eigenvalue / attractor / dynamical system
    (
        [
            "eigenvalue", "state-space", "state space",
            "attractor", "dynamical system", "dx/dt",
            "stability analysis",
            "perturbation recovery trajectory",
        ],
        "state_space",
    ),
    # Delay encoding / DenRAM / phase encoding
    (
        [
            "delay encoding", "phase encoding", "denram",
            "phase-delay", "phase delay", "static attractor",
            "cross-talk probability", "cross talk probability",
            "oscillatory timing",
        ],
        "denram",
    ),
    # Heterogeneity / variability / disorder resilience
    (
        [
            "heterogeneity", "heterogeneous substrate",
            "vm_rest variance", "vmem variance",
            "gap junction variance", "ion expression variability",
            "natural disorder", "distributed resilience",
            "attractor basin depth",
        ],
        "heterogeneity",
    ),
    # Homeostatic / BESS / scaling / adversarial recovery
    (
        [
            "homeostatic", "homeostasis", "bess upgrade",
            "scaling factor", "adversarial bias",
            "adversarial recovery", "repeated perturbation",
            "activity threshold",
        ],
        "homeostatic",
    ),
    # Model validation / sensitivity / falsification / competing models
    (
        [
            "sensitivity analysis", "parameter sensitivity",
            "model validation", "model comparison",
            "falsification", "falsifiable", "kill condition",
            "competing model", "scale-free", "scale free",
            "barabasi", "barabási", "preferential attachment",
            "nonlinear coupling", "voltage-dependent",
            "assumption", "assumptions at risk",
            "robust conclusion",
        ],
        "model_validation",
    ),
]


def detect_computation_query(query: str) -> list[str]:
    """Detect which reasoning engine computations the query requires.

    Returns a list of computation module keys (e.g., ["ribozyme", "freeze_thaw"]).
    """
    lower = query.lower()
    matched: list[str] = []
    for patterns, module_key in _COMPUTATION_TRIGGERS:
        for pattern in patterns:
            if pattern in lower:
                if module_key not in matched:
                    matched.append(module_key)
                break
    return matched


def _extract_query_params(query: str) -> dict:
    """Extract numeric parameters from a query for computation customization.

    Looks for patterns like:
        fidelity of 94.1%, 72-day cycle, 99.9% integrity, etc.
    """
    import re as _re
    params: dict = {}

    # Fidelity
    m = _re.search(r"fidelity\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%", query, _re.IGNORECASE)
    if m:
        params["fidelity_per_nt"] = float(m.group(1)) / 100.0

    # Copy cycle
    m = _re.search(r"(\d+)\s*-?\s*day\s*(?:generation\s*)?(?:cycle|period)", query, _re.IGNORECASE)
    if m:
        params["copy_cycle_days"] = float(m.group(1))

    # Data integrity / survival probability
    m = _re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:data\s*)?(?:integrity|survival|stability)", query, _re.IGNORECASE)
    if m:
        params["target_survival_prob"] = float(m.group(1)) / 100.0

    # Duration
    m = _re.search(r"(?:for\s+)?(\d+)\s*(?:-?\s*)year", query, _re.IGNORECASE)
    if m:
        params["target_duration_days"] = float(m.group(1)) * 365.25

    # Temperature
    m = _re.search(r"(-?\d+(?:\.\d+)?)\s*°?\s*C", query)
    if m:
        params["freeze_temp_C"] = float(m.group(1))

    # Node count
    m = _re.search(r"(\d+)\s*(?:nodes?|cells?|clusters?)", query, _re.IGNORECASE)
    if m:
        params["n_nodes"] = int(m.group(1))

    # Rewire budget
    m = _re.search(r"(\d+)\s*(?:rewir|edge)", query, _re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if val <= 50:  # sanity check — it's a budget, not a node count
            params["rewire_budget"] = val

    return params


def run_computation(modules: list[str], query: str) -> str:
    """Run reasoning engine computations and return formatted context.

    This is the computation analog of generate_fallback_context().
    """
    if not modules:
        return ""

    params = _extract_query_params(query)
    lines: list[str] = [
        "",
        "═══════════════════════════════════════════════════════════════════════",
        "REASONING ENGINE — PRE-COMPUTED RESULTS [SIMULATION-DERIVED]",
        "═══════════════════════════════════════════════════════════════════════",
        "",
    ]

    for module_key in modules:
        try:
            if module_key == "ribozyme":
                lines.extend(_compute_ribozyme(params))
            elif module_key == "freeze_thaw":
                lines.extend(_compute_freeze_thaw(params))
            elif module_key == "mosaic":
                lines.extend(_compute_mosaic(params))
            elif module_key == "spectral_rewiring":
                lines.extend(_compute_spectral_rewiring(params))
            elif module_key == "state_space":
                lines.extend(_compute_state_space(params))
            elif module_key == "denram":
                lines.extend(_compute_denram(params))
            elif module_key == "heterogeneity":
                lines.extend(_compute_heterogeneity(params))
            elif module_key == "homeostatic":
                lines.extend(_compute_homeostatic(params))
            elif module_key == "model_validation":
                lines.extend(_compute_model_validation(params))
        except Exception as exc:
            lines.append(f"[COMPUTATION ERROR: {module_key}] {exc}")
            lines.append("")

    lines.append("═══════════════════════════════════════════════════════════════════════")
    lines.append("USE these computed values directly. They are deterministic outputs from")
    lines.append("the Acheron Reasoning Engine. Tag as [SIMULATION-DERIVED].")
    lines.append("Present the ANSWER first, then show the derivation.")
    lines.append("═══════════════════════════════════════════════════════════════════════")
    lines.append("")

    return "\n".join(lines)


def _compute_ribozyme(params: dict) -> list[str]:
    """Run QT45 ribozyme computations."""
    from acheron.reasoning.ribozyme import (
        QT45Parameters,
        analyze_qt45,
        compute_redundancy_for_stability,
        compute_error_correction_overhead,
        compute_corruption_after_n_cycles,
    )
    import math

    fidelity = params.get("fidelity_per_nt", 0.941)
    cycle_days = params.get("copy_cycle_days", 72)
    target_prob = params.get("target_survival_prob", 0.999)
    duration_days = params.get("target_duration_days", 365.25)

    qt_params = QT45Parameters(fidelity_per_nt=fidelity, copy_cycle_days=cycle_days)
    redundancy = compute_redundancy_for_stability(qt_params, duration_days, target_prob)
    ecc = compute_error_correction_overhead(qt_params)

    n_cycles = math.ceil(duration_days / cycle_days)
    corruption = compute_corruption_after_n_cycles(
        n_cycles, qt_params, redundancy=max(1, redundancy.copies_needed),
    )

    lines = [
        "COMPUTATION: QT45 RIBOZYME / BIO-RAID REDUNDANCY",
        "─" * 60,
        "",
        "Input Parameters:",
        f"  Strand length:           {qt_params.length_nt} nt",
        f"  Fidelity per nucleotide: {qt_params.fidelity_per_nt}",
        f"  Copy cycle:              {qt_params.copy_cycle_days} days",
        f"  Yield per cycle:         {qt_params.yield_fraction * 100}%",
        f"  Target duration:         {duration_days:.1f} days ({duration_days / 365.25:.1f} years)",
        f"  Target survival P:       {target_prob}",
        "",
        "Derived Values:",
        f"  Per-copy fidelity (full strand): {qt_params.per_copy_fidelity:.6f}",
        f"  Per-copy error rate:             {qt_params.per_copy_error_rate:.6f}",
        f"  Copy cycles in target period:    {redundancy.n_copy_cycles}",
        f"  Copies per year:                 {qt_params.copies_per_year:.2f}",
        "",
        "RESULT — Redundancy Factor:",
        f"  Copies required:           {redundancy.copies_needed}",
        f"  Redundancy factor:         {redundancy.redundancy_factor}",
        f"  Actual survival P:         {redundancy.survival_probability_actual}",
        "",
        "Error Correction Overhead:",
        f"  Raw error rate per nt:     {ecc.raw_error_rate_per_nt}",
        f"  Parity nucleotides:        {ecc.parity_nucleotides_needed} / {qt_params.length_nt} nt",
        f"  Overhead fraction:         {ecc.overhead_fraction}",
        f"  Effective payload:         {ecc.effective_payload_nt} nt",
        f"  Correctable errors/copy:   {ecc.correctable_errors_per_copy}",
        "",
        f"Corruption after {n_cycles} cycles ({duration_days / 365.25:.1f} yr):",
        f"  P(single copy corrupt):    {corruption.p_corruption_single_copy}",
        f"  P(all copies corrupt):     {corruption.p_corruption_with_redundancy}",
        f"  Expected intact copies:    {corruption.expected_intact_copies}",
        "",
        "Derivation:",
        f"  P(perfect copy) = {fidelity}^{qt_params.length_nt} = {qt_params.per_copy_fidelity:.6f}",
        f"  P(lineage survives {n_cycles} cycles) = {qt_params.per_copy_fidelity:.6f}^{n_cycles}"
        f" = {qt_params.per_copy_fidelity ** n_cycles:.6e}",
        f"  M copies needed: log({1 - target_prob}) / log(1 - P_survive) = {redundancy.copies_needed}",
        "",
    ]
    return lines


def _compute_freeze_thaw(params: dict) -> list[str]:
    """Run freeze-thaw kinetic computations."""
    from acheron.reasoning.freeze_thaw import (
        FreezeThawParams,
        analyze_freeze_thaw,
    )

    ft_params = FreezeThawParams()
    if "freeze_temp_C" in params:
        ft_params.freeze_temp_C = params["freeze_temp_C"]

    result = analyze_freeze_thaw(ft_params)

    lines = [
        "COMPUTATION: FREEZE-THAW KINETICS",
        "─" * 60,
        "",
        f"  Freeze temperature:          {ft_params.freeze_temp_C}°C",
        f"  Thaw temperature:            {ft_params.thaw_temp_C}°C",
        f"  Stall probability:           {result.stall_probability}",
        f"  Strand separation @ freeze:  {result.strand_separation_at_freeze}",
        f"  Strand separation @ thaw:    {result.strand_separation_at_thaw}",
        f"  Concentration factor:        {result.concentration_factor}x",
        f"  Optimal freeze duration:     {result.optimal_interval.optimal_freeze_hours} h",
        f"  Optimal thaw duration:       {result.optimal_interval.optimal_thaw_hours} h",
        f"  Optimal cycle:               {result.optimal_interval.optimal_cycle_hours} h",
        f"  Replication efficiency:      {result.optimal_interval.replication_efficiency}",
        f"  Triplet trapping P:          {result.triplet_trapping.trapping_probability:.6f}",
        f"  Stalled triplets:            {result.triplet_trapping.stalled_triplets}/{result.triplet_trapping.n_triplets}",
        "",
    ]
    return lines


def _compute_mosaic(params: dict) -> list[str]:
    """Run topology comparison computations."""
    from acheron.reasoning.mosaic import compare_topologies

    n = params.get("n_nodes", 100)
    comp = compare_topologies(n_nodes=n, seed=42)

    lines = [
        f"COMPUTATION: TOPOLOGY COMPARISON ({n} nodes)",
        "─" * 60,
        "",
    ]
    for label, m in [
        ("Ring Lattice", comp.ring_lattice),
        ("Small World", comp.small_world),
        ("Uniform Random", comp.uniform),
    ]:
        lines.append(f"  {label}:")
        lines.append(f"    Clustering coefficient:     {m.clustering_coefficient}")
        lines.append(f"    Avg path length:            {m.average_path_length}")
        lines.append(f"    Signal latency:             {m.signal_propagation_latency_ms} ms")
        lines.append(f"    Energy per route:           {m.energy_per_routing_event_J:.2e} J")
        lines.append(f"    Error cascade P:            {m.error_cascade_probability}")
        lines.append(f"    Fault tolerance:            {m.fault_tolerance_fraction}")
        lines.append("")
    return lines


def _compute_spectral_rewiring(params: dict) -> list[str]:
    """Run spectral gap analysis + energy-optimized edge rewiring.

    Implements the Payvand Mosaic in-memory routing principle:
    - Compute graph Laplacian spectral gap (algebraic connectivity)
    - Find optimal edge rewirings that maximize spectral gap / energy ratio
    - Provide cross-species empirical grounding
    - Apply spectral threshold hypothesis model
    """
    from acheron.reasoning.mosaic import (
        build_small_world,
        analyze_spectral_properties,
        find_optimal_rewires,
        rewire_sweep,
    )
    from acheron.reasoning.empirical import (
        MOSAIC_SPEC,
        LEVIN_SPEC,
        SPECTRAL_THRESHOLD,
        EMPIRICAL_PARAMS,
        get_empirical_context,
        format_cross_species_table,
    )

    n = params.get("n_nodes", 50)
    k = 6
    rewire_budget = params.get("rewire_budget", 5)

    # Build baseline small-world graph
    graph = build_small_world(n, k, rewire_prob=0.1, seed=42)

    # Spectral analysis
    spectral = analyze_spectral_properties(graph)

    # Energy-optimized rewiring
    rewire = find_optimal_rewires(graph, rewire_budget=rewire_budget, seed=42)

    # Beta sweep
    sweep = rewire_sweep(n_nodes=min(n, 30), k_neighbors=k, seed=42)

    # Spectral threshold assessment
    stall_risk = SPECTRAL_THRESHOLD.is_stall_risk(spectral.spectral_gap)
    is_optimal = SPECTRAL_THRESHOLD.is_optimal(spectral.spectral_gap)
    ed_cost = SPECTRAL_THRESHOLD.energy_delay_cost(
        spectral.total_network_energy_J, spectral.spectral_gap,
    )

    lines = [
        f"COMPUTATION: SPECTRAL ANALYSIS + ENERGY-OPTIMIZED REWIRING ({n} nodes)",
        "─" * 60,
        "",
        "BASELINE SPECTRAL PROPERTIES:",
        f"  Spectral gap (lambda_2):      {spectral.spectral_gap}",
        f"  Laplacian eigenvalues (first): {spectral.laplacian_eigenvalues[:6]}",
        f"  Total network energy:          {spectral.total_network_energy_J:.4e} J",
        f"  Energy per edge:               {spectral.energy_per_edge_J:.4e} J",
        f"  Spectral efficiency:           {spectral.spectral_efficiency} (gap/energy)",
        "",
        "SPECTRAL THRESHOLD ASSESSMENT:",
        f"  Target lambda_2:               {SPECTRAL_THRESHOLD.theta_optimal} [BOUNDED-INFERENCE]",
        f"  Critical threshold:            {SPECTRAL_THRESHOLD.theta_critical}",
        f"  Current status:                {'OPTIMAL' if is_optimal else 'STALL RISK' if stall_risk else 'SUB-OPTIMAL'}",
        f"  Energy-Delay product:          {ed_cost:.6e}",
        "",
    ]

    # Rewiring results
    lines.extend([
        "ENERGY-OPTIMIZED EDGE REWIRING (Payvand Mosaic Principle):",
        f"  Rewire budget:                 {rewire.rewire_budget} edges",
        f"  Candidates evaluated:          {len(rewire.candidates)}",
        f"  Beneficial rewires found:      {len(rewire.optimal_rewires)}",
        "",
        f"  Baseline spectral gap:         {rewire.baseline_spectral_gap}",
        f"  Optimized spectral gap:        {rewire.optimized_spectral_gap}",
        f"  Gap improvement:               +{rewire.gap_improvement_pct}%",
        "",
        f"  Baseline energy:               {rewire.baseline_total_energy_J:.4e} J",
        f"  Optimized energy:              {rewire.optimized_total_energy_J:.4e} J",
        f"  Energy reduction:              {rewire.energy_reduction_pct}%",
        "",
        f"  Baseline efficiency:           {rewire.baseline_spectral_efficiency}",
        f"  Optimized efficiency:          {rewire.optimized_spectral_efficiency}",
        "",
    ])

    if rewire.optimal_rewires:
        lines.append("  OPTIMAL REWIRING EVENTS:")
        for i, rw in enumerate(rewire.optimal_rewires, 1):
            lines.extend([
                f"    #{i}: {rw.original_source}→{rw.original_target}  "
                f"REWIRE TO  {rw.new_source}→{rw.new_target}",
                f"        Conductance: {rw.original_conductance_nS} nS → {rw.new_conductance_nS} nS",
                f"        Δ spectral gap: +{rw.delta_spectral_gap}",
                f"        Δ energy: {rw.delta_energy_J:.4e} J",
                f"        Efficiency: {rw.efficiency_score}",
            ])
        lines.append("")

    # Beta sweep results
    lines.append("  WATTS-STROGATZ BETA SWEEP (spectral gap vs rewire probability):")
    for pt in sweep:
        marker = " ◄ optimal" if pt["spectral_efficiency"] == max(r["spectral_efficiency"] for r in sweep) else ""
        lines.append(
            f"    β={pt['rewire_prob']:.2f}  λ₂={pt['spectral_gap']:.6f}  "
            f"CC={pt['clustering_coefficient']:.4f}  "
            f"APL={pt['average_path_length']:.2f}  "
            f"eff={pt['spectral_efficiency']}{marker}"
        )
    lines.append("")

    # Payvand Mosaic reference
    lines.extend([
        "MOSAIC ARCHITECTURE REFERENCE (Payvand et al.):",
        f"  Architecture:        {MOSAIC_SPEC.architecture_type}",
        f"  Routing substrate:   {MOSAIC_SPEC.routing_substrate}",
        f"  Principle:           {MOSAIC_SPEC.routing_principle}",
        f"  Energy/route:        {MOSAIC_SPEC.energy_per_route_label}",
        f"  READ operation:      {MOSAIC_SPEC.read_operation}",
        f"  WRITE operation:     {MOSAIC_SPEC.write_operation}",
        f"  Tile model:          {MOSAIC_SPEC.tile_model}",
        "",
    ])

    # Bioelectric memory reference
    lines.extend([
        "BIOELECTRIC MEMORY REFERENCE (Levin et al.):",
        f"  T_hold (initial):    {LEVIN_SPEC.t_hold_initial_label}",
        f"  T_hold (steady):     {LEVIN_SPEC.t_hold_steady_state}",
        f"  Vmem swing:          {LEVIN_SPEC.vmem_delta_mV} mV ({LEVIN_SPEC.vmem_hyperpolarized_mV} to {LEVIN_SPEC.vmem_depolarized_mV} mV)",
        f"  GJ conductance:      {LEVIN_SPEC.gap_junction_conductance_label}",
        f"  GJ type (planarian): {LEVIN_SPEC.gap_junction_type_planarian}",
        "",
    ])

    # Cross-species summary (compact)
    lines.extend([
        "CROSS-SPECIES COMPARISON (EMPIRICALLY GROUNDED):",
        "  | System    | Vmem Ease | Persistence        | I/O Speed | GJ Type  | Spectral |",
        "  |-----------|-----------|--------------------|-----------|-----------| ---------|",
        "  | Planarian | High      | 3-6h / Permanent   | s-min     | Innexin   | HIGH    |",
        "  | Xenopus   | High      | hours-days / Perm   | ms-min    | Connexin  | MODERATE|",
        "  | Physarum  | Moderate  | hours-days          | s-min     | None/Tube | HIGH    |",
        "  | Organoid  | Low-Mod   | days-weeks          | ms-s      | Connexin  | LOW-MOD |",
        "",
    ])

    # Hypothesis + falsification
    lines.extend([
        "SPECTRAL THRESHOLD HYPOTHESIS:",
        f"  {SPECTRAL_THRESHOLD.theta_critical_label}",
        f"  Energy-delay model: {SPECTRAL_THRESHOLD.energy_delay_product_model}",
        f"  Success metric: {SPECTRAL_THRESHOLD.success_metric}",
        "",
        "KILL CONDITION:",
        f"  {SPECTRAL_THRESHOLD.kill_condition}",
        "",
    ])

    return lines


def _compute_state_space(params: dict) -> list[str]:
    """Run state-space dynamical analysis."""
    from acheron.reasoning.mosaic import build_small_world
    from acheron.reasoning.state_space import analyze_state_space

    n = params.get("n_nodes", 30)
    graph = build_small_world(n, 4, seed=42)
    result = analyze_state_space(graph)

    lines = [
        f"COMPUTATION: STATE-SPACE ANALYSIS ({n} nodes)",
        "─" * 60,
        "",
        f"  Dimension:          {result.system_dimension}",
        f"  Stable:             {result.stability.is_stable}",
        f"  Dominant eigenvalue: {result.stability.dominant_eigenvalue}",
        f"  Eigenvalues (top):  {result.stability.eigenvalues}",
        f"  Spectral gap:       {result.stability.spectral_gap}",
        f"  Convergence rate:   {result.stability.convergence_rate}",
    ]
    if result.attractors:
        lines.append(f"  Attractors found:   {len(result.attractors)}")
        lines.append(f"  Convergence steps:  {result.attractors[0].convergence_steps}")
    if result.recovery:
        lines.append(f"  Recovery steps:     {result.recovery.recovery_steps}")
        lines.append(f"  Recovery converged: {result.recovery.converged}")
    lines.append("")
    return lines


def _compute_denram(params: dict) -> list[str]:
    """Run delay encoding analysis."""
    from acheron.reasoning.denram import EncodingMode, analyze_delay_encoding

    phase = analyze_delay_encoding(EncodingMode.PHASE_DELAY, seed=42)
    static = analyze_delay_encoding(EncodingMode.STATIC_ATTRACTOR, seed=42)

    lines = [
        "COMPUTATION: DELAY ENCODING ANALYSIS",
        "─" * 60,
        "",
        "  Phase-Delay Mode:",
        f"    Stability score:         {phase.phase_stability_score}",
        f"    Mean persistence:        {phase.mean_persistence_ms} ms",
        f"    Noise tolerance:         {phase.noise_tolerance_mV} mV",
        f"    Cross-talk P:            {phase.cross_talk_probability}",
        f"    Min cluster separation:  {phase.min_cluster_separation_um} um",
        "",
        "  Static Attractor Mode:",
        f"    Stability score:         {static.phase_stability_score}",
        f"    Mean persistence:        {static.mean_persistence_ms} ms",
        f"    Cross-talk P:            {static.cross_talk_probability}",
        "",
    ]
    return lines


def _compute_heterogeneity(params: dict) -> list[str]:
    """Run heterogeneity comparison."""
    from acheron.reasoning.heterogeneity import compare_substrates

    n = params.get("n_nodes", 50)
    comp = compare_substrates(n_nodes=n, seed=42)

    lines = [
        f"COMPUTATION: HETEROGENEITY COMPARISON ({n} nodes)",
        "─" * 60,
        "",
        "  Uniform Substrate:",
        f"    Recovery steps:    {comp.uniform_result.recovery_steps}",
        f"    Recovery complete: {comp.uniform_result.recovery_complete}",
        f"    Residual error:    {comp.uniform_result.residual_error_mV} mV",
        f"    Basin depth:       {comp.uniform_basin_depth_mV} mV",
        "",
        "  Heterogeneous Substrate:",
        f"    Recovery steps:    {comp.heterogeneous_result.recovery_steps}",
        f"    Recovery complete: {comp.heterogeneous_result.recovery_complete}",
        f"    Residual error:    {comp.heterogeneous_result.residual_error_mV} mV",
        f"    Basin depth:       {comp.heterogeneous_basin_depth_mV} mV",
        "",
        f"  Conclusion: {comp.conclusion}",
        "",
    ]
    return lines


def _compute_homeostatic(params: dict) -> list[str]:
    """Run homeostatic recovery simulations."""
    from acheron.reasoning.mosaic import build_small_world
    from acheron.reasoning.homeostatic import (
        simulate_adversarial_recovery,
        simulate_repeated_perturbation,
    )

    graph = build_small_world(50, 6, seed=42)
    adv = simulate_adversarial_recovery(graph, bias_mV=30.0, seed=42)

    graph2 = build_small_world(50, 6, seed=42)
    rep = simulate_repeated_perturbation(graph2, n_perturbations=5, seed=42)

    lines = [
        "COMPUTATION: HOMEOSTATIC RECOVERY (50 nodes)",
        "─" * 60,
        "",
        "  Adversarial Bias (30 mV):",
        f"    Recovery steps:    {adv.recovery_steps}",
        f"    Recovery complete: {adv.recovery_complete}",
        f"    Final activity:    {adv.final_activity}",
        f"    Total energy:      {adv.total_energy_J:.2e} J",
        f"    Scaling events:    {adv.total_scaling_events}",
        "",
        "  Repeated Perturbation (5x):",
        f"    Stability held:    {rep.stability_maintained}",
        f"    Activity variance: {rep.activity_variance}",
        f"    Total energy:      {rep.total_energy_J:.2e} J",
        "",
    ]
    return lines


def _compute_model_validation(params: dict) -> list[str]:
    """Run sensitivity analysis + falsification registry."""
    from acheron.reasoning.sensitivity import (
        run_sensitivity_analysis,
        compare_topology_models,
    )
    from acheron.reasoning.falsification import (
        run_falsification_analysis,
        PredictionStatus,
    )

    n_nodes = params.get("n_nodes", 30)
    k_neighbors = params.get("k_neighbors", 4)
    seed = params.get("seed", 42)

    # Sensitivity analysis
    sens = run_sensitivity_analysis(n_nodes, k_neighbors, seed=seed)

    # Falsification registry
    fals = run_falsification_analysis(n_nodes, k_neighbors, seed=seed)

    # Cross-model comparison
    comparisons = compare_topology_models(n_nodes, k_neighbors, seed=seed)

    lines = [
        "COMPUTATION: MODEL VALIDATION META-LAYER",
        "─" * 60,
        "",
        "  ┌─ PARAMETER SENSITIVITY ANALYSIS ─┐",
        "",
    ]

    for s in sens.parameter_sensitivities:
        status = "ROBUST" if s.robust else "HIGH SENSITIVITY"
        lines.append(f"    {s.parameter_name} → {s.output_name}: "
                     f"S={s.sensitivity_index} [{status}]")
    lines.append("")

    lines.append("  ┌─ CROSS-MODEL TOPOLOGY COMPARISON ─┐")
    lines.append("")
    for mc in comparisons:
        lines.append(f"    {mc.metric_name}:")
        lines.append(f"      WS={mc.watts_strogatz}  BA={mc.barabasi_albert}  "
                     f"ER={mc.erdos_renyi}")
        lines.append(f"      Agreement: {mc.agreement:.0%}  Dominant: {mc.dominant_model}")
    lines.append("")

    lines.append("  ┌─ FALSIFICATION PREDICTION REGISTRY ─┐")
    lines.append(f"    Overall: {fals.overall_health} "
                 f"({fals.alive_count} alive, {fals.at_risk_count} at risk, "
                 f"{fals.falsified_count} falsified)")
    lines.append("")

    for pred in fals.predictions:
        icon = {
            PredictionStatus.ALIVE: "+",
            PredictionStatus.AT_RISK: "~",
            PredictionStatus.FALSIFIED: "X",
            PredictionStatus.UNTESTED: "?",
        }[pred.status]
        lines.append(f"    [{icon}] {pred.prediction_id}: "
                     f"computed={pred.computed_value:.4f} "
                     f"margin={pred.margin:.4f} [{pred.status.value}]")
    lines.append("")

    if sens.robust_conclusions:
        lines.append("  ┌─ ROBUST CONCLUSIONS ─┐")
        for c in sens.robust_conclusions:
            lines.append(f"    + {c}")
        lines.append("")

    if sens.assumptions_at_risk:
        lines.append("  ┌─ ASSUMPTIONS AT RISK ─┐")
        for a in sens.assumptions_at_risk:
            lines.append(f"    ! {a}")
        lines.append("")

    return lines


LOGIC_HASH = """\
═══════════════════════════════════════════════════════════════════════
ACHERON NEXUS — CANONICAL LOGIC HASH v1.0
═══════════════════════════════════════════════════════════════════════

TITAN-LOGIC PHYSICS CONSTRAINTS (MANDATORY):
┌─────────────────────────────────────────────────────────────────────┐
│ Nernst:     E_ion = (RT/zF) × ln([Ion]_out/[Ion]_in)                │
│ GHK:        Vm = (RT/F) × ln((P_K[K+]o + P_Na[Na+]o + P_Cl[Cl-]i)  │
│                              /(P_K[K+]i + P_Na[Na+]i + P_Cl[Cl-]o))│
│ Gibbs:      ΔG = -nFE  |  Stable if ΔG > 10kT (~25 kJ/mol)         │
│ Shannon:    C = B × log₂(1 + SNR)  |  H = log₂(N_states)           │
└─────────────────────────────────────────────────────────────────────┘
Any claim violating these equations is REJECTED.

MULTI-AGENT ARCHITECTURE:
• Scraper    — Data retrieval only, no interpretation, flags paywalls
• Physicist  — Enforces Nernst/GHK/Gibbs, rejects non-physical claims
• Theorist   — Shannon entropy, BER, channel capacity, noise margins
• Lab Agent  — Converts hypotheses to protocols with PASS/FAIL criteria

NO NUMERIC INVENTION RULE (ABSOLUTE):
• [MEASURED]           = Cited organism-specific data (PMID/DOI)
• [SIMULATION-DERIVED] = From validated simulation with stated params
• [BOUNDED-INFERENCE]  = Physics-constrained estimate, not biological fact
• [TRANSFER]           = Cross-species data, state source organism
• UNKNOWN              = No data exists → propose measurement

FALSIFICATION / KILL SWITCH (MANDATORY):
Every hypothesis MUST include a KILL CRITERIA:
"If [measurable threshold] is exceeded, ABANDON this approach."
No hypothesis without falsification path is valid.

100-CELL CONSENSUS RESULTS (SIMULATION-DERIVED):
┌─────────────────────────────────────────────────────────────────────┐
│ BER vs Cell Count:    10 cells/bit → BER < 10⁻³ (SNR=2.0)          │
│ Max Tolerable Noise:  8.0 mV (for 10 cells, ±10mV signal margin)   │
│ Bistability Required: Passive gap junctions FAIL (drift to rest)   │
│ Minimum 4-bit Memory: 40 cells total (10 per bit)                  │
└─────────────────────────────────────────────────────────────────────┘

HARDWARE BASELINES:
• Planarian:  Vmem -20 to -60 mV, Innexins, τ_stab ≈ 40ms
• Xenopus:    Vmem -50 to -80 mV, Connexins, better characterized

BIO-ISA: SET_BIT | READ_BIT | GATE | AUTH | QUARANTINE | REWRITE
═══════════════════════════════════════════════════════════════════════
"""

# ======================================================================
# FAST MODE - Lean prompts for <1 minute responses
# ======================================================================

FAST_DECISION_PROMPT = """\
You are Nexus, a bioelectric research engine. Answer the question directly.

""" + LOGIC_HASH + """

PARAMETER FALLBACK RULE (CRITICAL):
When the user asks for a specific parameter (Vmem, conductance, permeability, etc.)
and no organism-specific measurement exists in the sources:
1. Use the PARAMETER FALLBACK DATA provided in the context
2. Present values from related species with confidence levels
3. Provide your ESTIMATE for the target species
4. Label all estimates as [BOUNDED-INFERENCE]
5. Include the measurement protocol to obtain the actual value

NEVER say "UNKNOWN - needs measurement" without providing:
- Cross-species reference values
- A bounded estimate for the target species
- The specific experiment to measure it

REASONING ENGINE COMPUTATION RULE (CRITICAL):
When REASONING ENGINE — PRE-COMPUTED RESULTS appear in the context:
1. These are deterministic outputs from the Acheron Reasoning Engine
2. Present the computed ANSWER first on line 1, then show the derivation
3. Tag all computed values as [SIMULATION-DERIVED]
4. Use the computed results as primary evidence — they override absent literature
5. Still cite the underlying model parameters and their evidence labels
6. Generate at least ONE hypothesis about implications of the computed result

RULES:
- For YES/NO questions: Start with "VERDICT: YES" or "VERDICT: NO" or "VERDICT: CONDITIONAL"
- For calculations: Start with "ANSWER: [value with units]"
- For parameter requests: Start with "PARAMETER: [name]" and provide fallback table
- Tag claims: [EVIDENCE], [INFERENCE], [SPECULATION], [BOUNDED-INFERENCE], [SIMULATION-DERIVED]
- No invented numbers. Cite sources, use cross-species estimates with labels, or use
  Reasoning Engine computed values.

OUTPUT FORMAT (keep it short):
1. VERDICT: or ANSWER: or PARAMETER: (first line, mandatory)
2. CONFIDENCE: [0-100]
3. RATIONALE: 3-5 sentences max, use physics equations where applicable
4. KEY ASSUMPTIONS: bullet list with tags
5. KILL CRITERIA: specific threshold for abandoning approach

Never use an em dash (—). Use a comma, period, or "and"/"but" instead.
"""

FAST_QUERY_TEMPLATE = """\
Sources:
{context}

Question: {query}

Start with VERDICT: or ANSWER: on line 1. Be concise."""


def get_mode_prompt(mode: NexusMode, fast: bool = True) -> str:
    """Return the system prompt for the given mode.

    Args:
        mode: The NexusMode to get prompt for
        fast: If True, use lean prompts for faster responses (default True)
    """
    # Decision mode uses fast prompt by default for speed
    if mode == NexusMode.DECISION:
        return FAST_DECISION_PROMPT if fast else DECISION_PROMPT

    if mode == NexusMode.HYPOTHESIS:
        return HYPOTHESIS_PROMPT
    elif mode == NexusMode.SYNTHESIS:
        return SYNTHESIS_PROMPT
    elif mode == NexusMode.TUTOR:
        return TUTOR_PROMPT
    return EVIDENCE_PROMPT


def get_mode_query_template(mode: NexusMode, fast: bool = True, query: str = "") -> str:
    """Return the query template for the given mode.

    Args:
        mode: The NexusMode to get template for
        fast: If True, use lean templates for faster responses (default True)
        query: The user's query (used for parameter fallback and computation detection)
    """
    # Detect if this is a parameter query and generate fallback context
    fallback_context = ""
    if query:
        matched_params = detect_parameter_query(query)
        if matched_params:
            fallback_context = generate_fallback_context(matched_params)

    # Detect computable queries and run reasoning engine calculations
    computation_context = ""
    if query:
        matched_computations = detect_computation_query(query)
        if matched_computations:
            computation_context = run_computation(matched_computations, query)

    # Combined extra context (parameter fallback + computation results)
    extra_context = fallback_context + computation_context

    # Plain English layer — injected into ALL modes for accessibility
    plain_english_context = ""
    if query:
        from acheron.reasoning.research_questions import (
            is_casual_query,
            translate_query,
            format_translated_query,
            detect_research_category,
            get_plain_english_injection,
        )
        if is_casual_query(query):
            tq = translate_query(query)
            plain_english_context = (
                "\n" + format_translated_query(tq) + "\n"
                + get_plain_english_injection(tq.category)
            )
        else:
            cats = detect_research_category(query)
            if cats:
                plain_english_context = get_plain_english_injection(cats[0])

    if mode == NexusMode.DECISION:
        if fast:
            if extra_context or plain_english_context:
                return FAST_QUERY_TEMPLATE.replace(
                    "Sources:\n{context}",
                    "Sources:\n{context}\n" + extra_context + plain_english_context
                )
            return FAST_QUERY_TEMPLATE
        base_template = """\
Retrieved source passages from the bioelectricity and biomedical research corpus:

=== SOURCE PASSAGES ===
{context}
========================
""" + extra_context + plain_english_context + """
Decision query: {query}

Issue a VERDICT first. Follow the exact output structure from your system \
prompt (VERDICT → CONFIDENCE → RATIONALE → KEY EVIDENCE → KEY UNKNOWNS → \
ACTION → KILL CRITERIA → PIVOT). Do NOT produce a full report. \
Every claim must trace to a source number."""
        return base_template
    if mode == NexusMode.TUTOR:
        return """\
Retrieved source passages from the bioelectricity and biomedical research corpus:

=== SOURCE PASSAGES ===
{context}
========================
""" + extra_context + plain_english_context + """
Learning query: {query}

Provide a technically accurate answer, then include the CONCEPTS FOR THE \
RESEARCHER section with: GLOSSARY (3 terms), ANALOGY (computing/cybersecurity), \
THE 'WHY' (connection to Acheron), and SELF-STUDY TASK (one specific resource). \
End with 2-3 FOLLOW-UP QUESTIONS. Every claim must trace to a source number.

IMPORTANT: If the sources do not contain the specific numerical values requested,
use the PARAMETER FALLBACK DATA above to provide cross-species estimates.
Always label these as [BOUNDED-INFERENCE] and state the measurement protocol
to obtain the actual value for the target species."""

    return """\
Retrieved source passages from the bioelectricity and biomedical research corpus:

=== SOURCE PASSAGES ===
{context}
========================
""" + extra_context + plain_english_context + """
Research query: {query}

Execute the full {mode_label} analysis using ALL the mandatory sections described \
in your system prompt. Be precise. Every claim must trace to a source number.

IMPORTANT: If the sources do not contain the specific numerical values requested,
use the PARAMETER FALLBACK DATA above to provide cross-species estimates.
Always label these as [BOUNDED-INFERENCE] and state the measurement protocol
to obtain the actual value for the target species.""".format(
        context="{context}",
        query="{query}",
        mode_label=mode.value.upper(),
    )


# ======================================================================
# Output parsing — extract structured data from LLM response
# ======================================================================

def parse_evidence_graph(raw_output: str) -> EvidenceGraph:
    """Parse EVIDENCE CLAIMS and CLAIM RELATIONSHIPS from LLM output."""
    claims: list[EvidenceClaim] = []
    edges: list[EvidenceEdge] = []

    claim_id_counter = 0

    # Parse claims
    in_claims = False
    in_relationships = False
    current_claim: dict = {}

    for line in raw_output.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()

        # Section detection
        if ("EVIDENCE CLAIM" in upper or "EVIDENCE EXTRACTED" in upper) and not in_relationships:
            in_claims = True
            in_relationships = False
            continue
        if "CLAIM RELATIONSHIP" in upper:
            # Flush pending claim
            if current_claim:
                claims.append(_build_claim(current_claim, claim_id_counter))
                claim_id_counter += 1
                current_claim = {}
            in_claims = False
            in_relationships = True
            continue
        if any(
            header in upper
            for header in [
                "AGREEMENT SUMMARY", "UNCERTAINTY", "HYPOTHES",
                "CONFIDENCE:", "NEXT_QUER", "SYSTEM DESIGN",
                "VALIDATION PATH", "OVERALL_CONFIDENCE",
                "SUBSTRATE SELECTION", "BIGR",
                "PROTOCOL SPECIFICATION", "FAULT TOLERANCE",
                "HEURISTIC BASELINE", "STRATEGIC RECOMMEND",
                "BIM QUANTIFICATION", "BIM SPECIFICATION",
                "GRAPH TOPOLOGY",
                "HARDWARE SPEC",
                "FALSIFICATION PROTOCOL", "KILL CONDITION",
                "SILICON-TO-CARBON", "ISA COMMAND",
                "BOOT PROTOCOL",
            ]
        ):
            if current_claim:
                claims.append(_build_claim(current_claim, claim_id_counter))
                claim_id_counter += 1
                current_claim = {}
            in_claims = False
            in_relationships = False
            continue

        if not stripped:
            continue

        # Parse claim fields
        if in_claims:
            if stripped.startswith("- CLAIM:") or stripped.startswith("CLAIM:"):
                if current_claim:
                    claims.append(_build_claim(current_claim, claim_id_counter))
                    claim_id_counter += 1
                current_claim = {"text": stripped.split(":", 1)[1].strip()}
            elif stripped.startswith("REFS:") or stripped.startswith("- REFS:"):
                current_claim["refs"] = re.findall(r"\[(\d+)\]", stripped)
            elif "SUPPORT_COUNT" in upper:
                m = re.search(r"(\d+)", stripped)
                if m:
                    current_claim["support"] = int(m.group(1))
            elif "CONTRADICTION_COUNT" in upper:
                m = re.search(r"(\d+)", stripped)
                if m:
                    current_claim["contradiction"] = int(m.group(1))
            elif "STATUS:" in upper:
                for status in ["supported", "mixed", "unclear", "unsupported"]:
                    if status in stripped.lower():
                        current_claim["status"] = status
                        break
            elif "STUDY_TYPE" in upper:
                current_claim["study_types"] = [
                    t.strip() for t in stripped.split(":", 1)[1].split(",") if t.strip()
                ] if ":" in stripped else []

        # Parse edges
        if in_relationships:
            for rel in ["supports", "contradicts", "depends-on", "depends on"]:
                if rel in stripped.lower():
                    parts = re.split(rf"\s+{rel}\s+", stripped.lstrip("- "), flags=re.IGNORECASE)
                    if len(parts) == 2:
                        edges.append(
                            EvidenceEdge(
                                source_claim=parts[0].strip(),
                                target_claim=parts[1].strip(),
                                relation=rel.replace(" ", "-"),
                            )
                        )
                    break

    # Flush final claim
    if current_claim:
        claims.append(_build_claim(current_claim, claim_id_counter))

    return EvidenceGraph(claims=claims, edges=edges)


def _build_claim(data: dict, idx: int) -> EvidenceClaim:
    """Build an EvidenceClaim from parsed fields."""
    text = data.get("text", "")
    # Try to split into subject-predicate-object
    subject, predicate, obj = _parse_spo(text)

    support = data.get("support", 0)
    contradiction = data.get("contradiction", 0)
    total = support + contradiction
    agreement = support / total if total > 0 else 0.0

    status_str = data.get("status", "unclear")
    try:
        status = ClaimStatus(status_str)
    except ValueError:
        status = ClaimStatus.UNCLEAR

    return EvidenceClaim(
        claim_id=f"C{idx}",
        subject=subject,
        predicate=predicate,
        object=obj,
        source_refs=[f"[{r}]" for r in data.get("refs", [])],
        support_count=support,
        contradiction_count=contradiction,
        agreement_score=agreement,
        status=status,
        study_types=data.get("study_types", []),
    )


def _parse_spo(text: str) -> tuple[str, str, str]:
    """Try to parse subject–predicate–object from a claim text.

    Falls back to (full_text, '', '') if parsing fails.
    """
    # Try splitting on common predicate verbs
    for pattern in [
        r"^(.+?)\s+(regulates|controls|inhibits|activates|modulates|causes|"
        r"induces|affects|drives|encodes|maintains|signals|mediates|"
        r"promotes|suppresses|blocks|enables|requires)\s+(.+)$",
    ]:
        m = re.match(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

    # Try splitting on "is" / "are"
    m = re.match(r"^(.+?)\s+(is|are)\s+(.+)$", text, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

    return text, "", ""


def parse_hypotheses(raw_output: str) -> list[RankedHypothesis]:
    """Parse HYPOTHESES section from LLM output into ranked hypotheses."""
    hypotheses: list[RankedHypothesis] = []
    current: dict = {}
    in_hypotheses = False

    for line in raw_output.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()

        # Detect start of hypotheses section header (but not individual HYPOTHESIS: entries)
        if "HYPOTHES" in upper and (
            upper.startswith("HYPOTHES") or upper.startswith("#")
            or upper.startswith("*") or upper.startswith("2.")
            or upper.startswith("4.")
        ) and not stripped.startswith("HYPOTHESIS:") and not stripped.startswith("- HYPOTHESIS:"):
            in_hypotheses = True
            continue

        # Detect end of hypotheses section
        if in_hypotheses and any(
            header in upper
            for header in [
                "UNCERTAINTY", "OVERALL_CONFIDENCE", "NEXT_QUER",
                "SYSTEM DESIGN", "VALIDATION PATH", "BIOELECTRIC SCHEMATIC",
                "SUBSTRATE SELECTION", "PROTOCOL SPECIFICATION",
                "FAULT TOLERANCE", "STRATEGIC RECOMMEND",
                "BIM QUANTIFICATION", "BIM SPECIFICATION",
                "GRAPH TOPOLOGY", "TRANSFER LOGIC",
                "HARDWARE SPEC", "NEXT COLLECTION",
            ]
        ):
            if current:
                hypotheses.append(_build_hypothesis(current, len(hypotheses)))
                current = {}
            in_hypotheses = False
            continue

        if not in_hypotheses or not stripped:
            continue

        # Track which sub-section of a hypothesis we're accumulating into.
        # This supports both old structured format and new plain-English format.

        # --- New hypothesis entry ---
        if (
            stripped.startswith("HYPOTHESIS:")
            or stripped.startswith("- HYPOTHESIS:")
            or upper == "HYPOTHESIS:"
        ):
            if current:
                hypotheses.append(_build_hypothesis(current, len(hypotheses)))
            stmt = stripped.split(":", 1)[1].strip()
            current = {"statement": stmt, "_section": "" if stmt else "rationale"}
        elif re.match(r"^Hypothesis:\s*$", stripped):
            # v3 format: "Hypothesis:" on its own line, body follows
            if current:
                hypotheses.append(_build_hypothesis(current, len(hypotheses)))
            current = {"statement": "", "_section": "rationale"}
        elif stripped.startswith("[H") or re.match(r"^H\d+[:\.]", stripped):
            if current:
                hypotheses.append(_build_hypothesis(current, len(hypotheses)))
            current = {
                "statement": re.sub(r"^\[?H\d+\]?[:\.\s]*", "", stripped).strip(),
                "_section": "",
            }

        # --- v3 Discovery Engine section headers ---
        elif "THIS HYPOTHESIS IS BASED ON" in upper:
            current["_section"] = "evidence_for"
            current.setdefault("predictions", [])
        elif "PREDICTED OBSERVABLE" in upper:
            current["_section"] = "predictions_section"
            current.setdefault("predictions", [])
        elif "EXPERIMENT PROPOSAL" in upper:
            current["_section"] = "minimal_test"
        elif upper.startswith("SIMULATION:") or "SIMULATION STEP" in upper:
            current["_section"] = "minimal_test"
        elif (
            upper.startswith("WET LAB:")
            or "WET-LAB STEP" in upper
            or "WET LAB STEP" in upper
        ):
            current["_section"] = "minimal_test"
        elif "PHASE-0" in upper or "PHASE 0" in upper:
            current["_section"] = "minimal_test"
        elif "PHASE-1" in upper or "PHASE 1" in upper:
            current["_section"] = "minimal_test"
        elif "TRANSFER LOGIC" in upper:
            current["_section"] = "known_unknowns"
            current.setdefault("known_unknowns", [])
        elif "CLOSED-LOOP TASK" in upper or "CLOSED LOOP TASK" in upper:
            current["_section"] = "minimal_test"
        elif "WET-LAB EXECUTION" in upper or "WET LAB EXECUTION" in upper:
            current["_section"] = "minimal_test"
        elif "HARDWARE REQUIREMENT" in upper:
            current["_section"] = "minimal_test"
        elif "REAGENT" in upper and "DYE" in upper:
            current["_section"] = "minimal_test"
        elif "SENSOR INTEGRATION" in upper:
            current["_section"] = "minimal_test"
        elif "QUANTIFICATION PLAN" in upper:
            current["_section"] = "minimal_test"
        elif "SUCCESS METRIC" in upper:
            current["_section"] = "minimal_test"
        elif "KILL CONDITION" in upper:
            current["_section"] = "minimal_test"
        elif "ISA COMMAND" in upper or "SILICON-TO-CARBON" in upper:
            current["_section"] = "minimal_test"
        elif "BOOT PROTOCOL" in upper:
            current["_section"] = "minimal_test"
        elif "DATA COLLECTION PLAN" in upper:
            current["_section"] = "minimal_test"
        elif "REFINEMENT:" in upper:
            current["_section"] = "minimal_test"

        # --- v2 Plain-English section headers (backward-compatible) ---
        elif "THE IDEA IN PLAIN ENGLISH" in upper:
            current["_section"] = "rationale"
        elif "WHAT MUST BE TRUE" in upper:
            current["_section"] = "assumptions"
            current.setdefault("assumptions", [])
        elif (
            "WHAT WE KNOW" in upper
            and "DON'T" not in upper
            and "DON\u2019T" not in upper
            and "NOT" not in upper
        ):
            current["_section"] = "evidence_for"
            current.setdefault("predictions", [])
        elif "WHAT WE ALREADY HAVE EVIDENCE" in upper:
            current["_section"] = "evidence_for"
            current.setdefault("predictions", [])
        elif (
            "WHAT WE DON'T KNOW" in upper
            or "WHAT WE DON\u2019T KNOW" in upper
        ):
            current["_section"] = "known_unknowns"
            current.setdefault("known_unknowns", [])
        elif "WHAT IS STILL UNKNOWN" in upper:
            current["_section"] = "known_unknowns"
            current.setdefault("known_unknowns", [])
        elif "WHAT THIS PREDICTS" in upper:
            current["_section"] = "predictions_section"
            current.setdefault("predictions", [])
        elif (
            "PHASE-0 EXPERIMENT" in upper
            or "PHASE 0 EXPERIMENT" in upper
        ):
            current["_section"] = "minimal_test"
        elif "FIRST TEST TO VALIDATE" in upper or "FIRST TEST:" in upper:
            current["_section"] = "minimal_test"
        elif "WHAT RESULT WOULD PROVE IT WRONG" in upper:
            current["_section"] = "falsifiers"
            current.setdefault("falsifiers", [])
        elif "NEXT 5 QUESTIONS" in upper or "NEXT FIVE QUESTIONS" in upper:
            current["_section"] = "next_questions"
            current.setdefault("next_questions", [])

        # --- Old structured format fields (backward-compatible) ---
        elif "RANK:" in upper:
            m = re.search(r"(\d+)", stripped)
            if m:
                current["rank"] = int(m.group(1))
        elif "EXPLANATORY_POWER:" in upper:
            m = re.search(r"([\d.]+)", stripped)
            if m:
                current["explanatory_power"] = float(m.group(1))
        elif "SIMPLICITY:" in upper:
            m = re.search(r"([\d.]+)", stripped)
            if m:
                current["simplicity"] = float(m.group(1))
        elif "CONSISTENCY:" in upper:
            m = re.search(r"([\d.]+)", stripped)
            if m:
                current["consistency"] = float(m.group(1))
        elif "MECHANISTIC_PLAUSIBILITY:" in upper:
            m = re.search(r"([\d.]+)", stripped)
            if m:
                current["mechanistic_plausibility"] = float(m.group(1))
        elif "RATIONALE:" in upper:
            current["rationale"] = stripped.split(":", 1)[1].strip()
        elif (
            "PREDICTION" in upper and ":" in stripped
            and current.get("_section") != "evidence_for"
        ):
            preds = current.get("predictions", [])
            val = stripped.split(":", 1)[1].strip()
            if val:
                preds.append(val)
            current["predictions"] = preds
        elif "FALSIF" in upper and ":" in stripped and current.get("_section") != "falsifiers":
            falsif = current.get("falsifiers", [])
            val = stripped.split(":", 1)[1].strip()
            if val:
                falsif.append(val)
            current["falsifiers"] = falsif
        elif "MINIMAL_TEST:" in upper:
            current["minimal_test"] = stripped.split(":", 1)[1].strip()
        elif upper.startswith("CONFIDENCE:") and "JUSTIFICATION" not in upper:
            m = re.search(r"(\d+)", stripped)
            if m:
                current["confidence"] = int(m.group(1))
        elif "CONFIDENCE_JUSTIFICATION:" in upper:
            current["confidence_justification"] = stripped.split(":", 1)[1].strip()
        elif "ASSUMPTION" in upper and ":" in stripped and current.get("_section") != "assumptions":
            assumptions = stripped.split(":", 1)[1].strip()
            current["assumptions"] = [
                a.strip() for a in re.split(r"[;,]|A\d+\.?\s*", assumptions) if a.strip()
            ]
        elif "KNOWN_UNKNOWN" in upper and ":" in stripped:
            unknowns = stripped.split(":", 1)[1].strip()
            current["known_unknowns"] = [u.strip() for u in unknowns.split(";") if u.strip()]
        elif "FAILURE_MODE" in upper and ":" in stripped:
            modes = stripped.split(":", 1)[1].strip()
            current["failure_modes"] = [m.strip() for m in modes.split(";") if m.strip()]
        elif upper.startswith("REFS:") or upper.startswith("- REFS:"):
            current["refs"] = re.findall(r"\[(\d+)\]", stripped)

        # --- Accumulate content into the current sub-section ---
        elif current and "statement" in current:
            section = current.get("_section", "")
            # Bullet or numbered line
            content_line = ""
            if stripped.startswith("- "):
                content_line = stripped[2:].strip()
            elif re.match(r"^(?:Step\s+)?\d+[.:)\s]", stripped):
                content_line = re.sub(r"^(?:Step\s+)?\d+[.:)\s]+", "", stripped).strip()
            elif stripped.startswith("* "):
                content_line = stripped[2:].strip()

            if section == "rationale":
                # Accumulate multi-line rationale
                existing = current.get("rationale", "")
                current["rationale"] = (existing + " " + stripped).strip() if existing else stripped
            elif section == "assumptions" and content_line:
                current.setdefault("assumptions", []).append(content_line)
            elif section == "evidence_for" and content_line:
                # Store evidence items; also extract refs
                current.setdefault("predictions", []).append(content_line)
                found_refs = re.findall(r"\[(\d+)\]", content_line)
                if found_refs:
                    current.setdefault("refs", []).extend(found_refs)
            elif section == "known_unknowns" and content_line:
                current.setdefault("known_unknowns", []).append(content_line)
            elif section == "predictions_section" and content_line:
                current.setdefault("predictions", []).append(content_line)
            elif section == "minimal_test":
                # Accumulate test steps
                step = content_line or stripped
                existing = current.get("minimal_test", "")
                current["minimal_test"] = (existing + "; " + step).strip("; ") if existing else step
            elif section == "falsifiers" and content_line:
                current.setdefault("falsifiers", []).append(content_line)
            elif section == "next_questions" and content_line:
                current.setdefault("next_questions", []).append(content_line)
            elif section == "":
                if stripped.startswith("- "):
                    # Fallback: try to assign to the most recent list field
                    val = stripped.lstrip("- ")
                    for field in ["predictions", "falsifiers", "assumptions",
                                  "known_unknowns", "failure_modes"]:
                        if field in current and isinstance(current[field], list):
                            current[field].append(val)
                            break
                else:
                    # v3 format: free-form text after HYPOTHESIS: is rationale
                    existing = current.get("rationale", "")
                    current["rationale"] = (
                        (existing + " " + stripped).strip()
                        if existing else stripped
                    )

    # Flush final hypothesis
    if current:
        hypotheses.append(_build_hypothesis(current, len(hypotheses)))

    return hypotheses


def _build_hypothesis(data: dict, idx: int) -> RankedHypothesis:
    """Build a RankedHypothesis from parsed fields.

    Handles both old structured format and new plain-English format.
    In the new format, "Next 5 Questions" are folded into known_unknowns
    since they represent the biggest uncertainties to resolve.
    """
    ep = min(data.get("explanatory_power", 0.5), 1.0)
    simp = min(data.get("simplicity", 0.5), 1.0)
    cons = min(data.get("consistency", 0.5), 1.0)
    mech = min(data.get("mechanistic_plausibility", 0.5), 1.0)
    overall = (ep * 0.3 + simp * 0.2 + cons * 0.3 + mech * 0.2)

    # Merge next_questions into known_unknowns (both represent uncertainties)
    known_unknowns = data.get("known_unknowns", [])
    next_questions = data.get("next_questions", [])
    if next_questions:
        known_unknowns = known_unknowns + next_questions

    return RankedHypothesis(
        hypothesis_id=f"H{idx + 1}",
        statement=data.get("statement", ""),
        rank=data.get("rank", idx + 1),
        explanatory_power=ep,
        simplicity=simp,
        consistency=cons,
        mechanistic_plausibility=mech,
        overall_score=round(overall, 3),
        rationale=data.get("rationale", ""),
        predictions=data.get("predictions", []),
        falsifiers=data.get("falsifiers", []),
        minimal_test=data.get("minimal_test", ""),
        confidence=min(data.get("confidence", 0), 100),
        confidence_justification=data.get("confidence_justification", ""),
        assumptions=data.get("assumptions", []),
        known_unknowns=known_unknowns,
        failure_modes=data.get("failure_modes", []),
        supporting_refs=[f"[{r}]" for r in data.get("refs", [])],
    )


def parse_next_queries(raw_output: str) -> list[str]:
    """Parse NEXT_QUERIES section from LLM output."""
    queries: list[str] = []
    in_section = False

    for line in raw_output.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()

        if "NEXT_QUER" in upper or "NEXT BEST EVIDENCE" in upper:
            in_section = True
            continue
        if in_section and any(
            header in upper
            for header in ["CONFIDENCE", "UNCERTAINTY", "HYPOTHES", "EVIDENCE"]
        ):
            break

        if in_section and stripped.startswith("- "):
            query = stripped.lstrip("- ").strip('"').strip("'")
            if query and len(query) > 5:
                queries.append(query)

    return queries


def parse_overall_confidence(raw_output: str) -> tuple[int, str]:
    """Parse OVERALL_CONFIDENCE and OVERALL_JUSTIFICATION."""
    confidence = 0
    justification = ""

    for line in raw_output.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()

        if "OVERALL_CONFIDENCE:" in upper or (
            "CONFIDENCE:" in upper and "OVERALL" in upper
        ):
            m = re.search(r"(\d+)", stripped)
            if m:
                confidence = min(int(m.group(1)), 100)
        elif "OVERALL_JUSTIFICATION:" in upper or "JUSTIFICATION:" in upper:
            justification = stripped.split(":", 1)[1].strip()

    return confidence, justification


def parse_uncertainty_notes(raw_output: str) -> list[str]:
    """Parse UNCERTAINTY section."""
    notes: list[str] = []
    in_section = False

    for line in raw_output.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()

        if "UNCERTAINTY" in upper and (
            upper.startswith("UNCERTAINTY") or upper.startswith("#")
            or upper.startswith("*") or "8." in upper
        ):
            in_section = True
            continue
        if in_section and any(
            header in upper
            for header in [
                "CONFIDENCE:", "NEXT_QUER", "OVERALL_CONFIDENCE",
                "HYPOTHES", "SYSTEM DESIGN",
            ]
        ):
            break

        if in_section and stripped.startswith("- "):
            notes.append(stripped.lstrip("- "))

    return notes


def parse_verdict(raw_output: str) -> tuple[str, str]:
    """Extract VERDICT line from decision-mode output.

    Returns (verdict, rationale) where verdict is YES/NO/CONDITIONAL
    and rationale is the text following the verdict keyword.
    If no verdict found, returns ("", "").
    """
    for line in raw_output.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("VERDICT:"):
            body = stripped.split(":", 1)[1].strip()
            # Extract the keyword
            for keyword in ("YES", "NO", "CONDITIONAL"):
                if keyword in body.upper():
                    # Rationale is the rest after the keyword
                    idx = body.upper().index(keyword) + len(keyword)
                    rationale = body[idx:].lstrip(" .,—-:").strip()
                    return keyword, rationale
            # Verdict line exists but no recognized keyword
            return body, ""
    return "", ""


def parse_answer(raw_output: str) -> tuple[str, str]:
    """Extract ANSWER line from calculation-mode output.

    Returns (answer, details) where answer is the numeric value/result
    and details is any additional text on the line.
    If no answer found, returns ("", "").
    """
    for line in raw_output.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("ANSWER:"):
            body = stripped.split(":", 1)[1].strip()
            # The answer is the full value (may include units)
            return body, ""
    return "", ""


def parse_parameter(raw_output: str) -> tuple[str, str]:
    """Extract PARAMETER line from parameter-query output.

    Returns (parameter_name, value) where parameter_name is what was asked
    and value is the estimate or table reference.
    If no parameter found, returns ("", "").
    """
    for line in raw_output.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("PARAMETER:"):
            body = stripped.split(":", 1)[1].strip()
            return body, ""
    return "", ""


def validate_decision_output(raw_output: str) -> list[str]:
    """Validate that decision-mode output contains a proper verdict, answer, or parameter.

    Decision mode now handles three types of questions:
    1. Verdict questions (should I, is it viable) → VERDICT: YES/NO/CONDITIONAL
    2. Calculation questions (what is the max, how many) → ANSWER: [value]
    3. Parameter questions (what is the conductance) → PARAMETER: [name]

    Returns a list of warnings (empty if output passes validation).
    """
    warnings: list[str] = []
    verdict, _ = parse_verdict(raw_output)
    answer, _ = parse_answer(raw_output)
    parameter, _ = parse_parameter(raw_output)

    # VERDICT, ANSWER, or PARAMETER is acceptable
    if not verdict and not answer and not parameter:
        warnings.append(
            "[VALIDATION] Decision mode was requested but no VERDICT:, ANSWER:, or PARAMETER: line "
            "was found in the output. The model may have fallen back to "
            "report-template behavior."
        )
    elif verdict and verdict not in ("YES", "NO", "CONDITIONAL"):
        warnings.append(
            f"[VALIDATION] VERDICT value '{verdict}' is not one of "
            "YES / NO / CONDITIONAL."
        )

    upper = raw_output.upper()
    # Kill criteria not required for parameter queries (they're informational)
    if "KILL CRITERIA" not in upper and not parameter:
        warnings.append(
            "[VALIDATION] No KILL CRITERIA section found. Decision output "
            "should specify when to abandon the approach."
        )
    return warnings


def build_engine_result(
    raw_output: str,
    query: str,
    mode: NexusMode,
    sources: list[QueryResult],
    total_searched: int,
    model_used: str,
    live_sources_fetched: int = 0,
) -> HypothesisEngineResult:
    """Parse the full LLM output into a HypothesisEngineResult."""
    evidence_graph = parse_evidence_graph(raw_output)
    evidence_graph.query = query

    hypotheses = []
    if mode in (NexusMode.HYPOTHESIS, NexusMode.SYNTHESIS):
        hypotheses = parse_hypotheses(raw_output)
        # Sort by overall score descending
        hypotheses.sort(key=lambda h: h.overall_score, reverse=True)
        # Re-assign ranks after sorting
        for i, h in enumerate(hypotheses):
            h.rank = i + 1

    next_queries = parse_next_queries(raw_output)
    confidence, justification = parse_overall_confidence(raw_output)
    uncertainty = parse_uncertainty_notes(raw_output)

    # Decision-mode output validation
    if mode == NexusMode.DECISION:
        validation_warnings = validate_decision_output(raw_output)
        if validation_warnings:
            uncertainty = validation_warnings + uncertainty
            logger.warning(
                "Decision output validation: %s", "; ".join(validation_warnings)
            )

    return HypothesisEngineResult(
        query=query,
        mode=mode,
        evidence_graph=evidence_graph,
        hypotheses=hypotheses,
        next_queries=next_queries,
        confidence=confidence,
        confidence_justification=justification,
        uncertainty_notes=uncertainty,
        sources=sources,
        model_used=model_used,
        total_chunks_searched=total_searched,
        live_sources_fetched=live_sources_fetched,
        raw_output=raw_output,  # Preserve the full LLM response
    )
