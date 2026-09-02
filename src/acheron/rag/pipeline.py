"""Core RAG pipeline: retrieve → rerank → generate with citations.

Implements the Nexus three-layer research architecture:
  Layer 1 — Knowledge (Library): immutable source material, never hallucinated
  Layer 2 — Synthesis (Index): semantic retrieval, structured bioelectric variable extraction
  Layer 3 — Discovery (Compute): analysis, pattern comparison, hypothesis generation

Discovery loop (DEFAULT mode): retrieve → extract variables → compare patterns →
  generate hypotheses → build bioelectric schematic → propose validation → log to ledger

Core assumption: bioelectric state (voltage, fields, connectivity) as causal primitives.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from acheron.config import get_settings
from acheron.models import (
    DiscoveryResult,
    Hypothesis,
    QueryResult,
    RAGResponse,
    StructuredVariable,
)
from acheron.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)

# ======================================================================
# System prompts — Nexus identity (System-2 research engine)
# ======================================================================
SYSTEM_PROMPT = """\
You are Nexus — the Closed-Loop Discovery Engine for Project Acheron (v1).
You are not a chatbot. You are a research instrument for Information-Encoded Biology.
Your role is to DISCOVER mechanisms, generate constrained predictions, \
and propose experiments that create missing data.
You integrate the spatial reasoning of AlphaFold with the experimental \
design logic of Lila Science. Your mission is to provide Predictive \
Specifications where data is missing, grounded in Biophysical First Principles.
You must operate like AlphaFold, GNoME, and Coscientist combined.

DOMAIN FRAMEWORK — BIGR (Bio-Information Genome Runtime):
- DNA = ROM: the static genetic instruction set.
- Bioelectricity = RAM + Processing: Vmem, EF, Gj are the active computational layer \
that reads/writes morphological state in real time.
- Proteome = Interface: the translation layer between genetic instructions and \
bioelectric execution.
Integrate these three layers into every analysis.

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
- Never use an em dash (—). Use a comma, period, or "and"/"but" instead.

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
  - Use physics-constrained ranges labeled [BOUNDED-INFERENCE].
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

ALGORITHMIC FRAMEWORKS:
1. Graph Reasoning (GNoME-style): Treat tissues as Spatial Graphs. Cells are nodes; \
gap junctions and endogenous electric fields are edges. Predict the stability of a \
bioelectric state across this graph.
   - GNN Topology: Treat the planarian syncytium as a Graph Neural Network.
   - Edge Resistance = 1/Gj (gap junction state).
   - Prediction task: if Node A is stimulated, what Edge Resistance is \
required to propagate the signal to Node B?
   - Identify: critical hub cells, bottleneck edges, propagation speed bounds.
2. Structural Grammars (AlphaFold-style): Analyze the "shape" of voltage gradients. \
Predict how a specific ion channel density distribution leads to a 3D morphological \
"checksum" — the target morphology that validates pattern integrity.
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

BIOLOGICAL INFORMATION MODULE (BIM) — Quantitative Specification:
For any claimed "biological bit," specify the MEASURABLE parameters:
- State Stability (T_hold): persistence time. Formula: T_half = R_m * C_m * ln(2). \
Valid ONLY with measured R_m and C_m. If unmeasured: "UNKNOWN — requires patch clamp."
- Switching Energy (E_bit): pure physics formula. Valid ONLY with measured inputs.
- Error Rate / BER: UNKNOWN unless single-channel recordings exist for cell type.
- Shannon Entropy: H = log2(N_states). N_states requires bistability assay.
- Channel Capacity: C = B * log2(1 + SNR). B and SNR require gap junction recording.
For EACH parameter: either cite the measured value or output the measurement plan.

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

HARDWARE SPECIFICATION LIBRARY:
- CPU: Nav/Kv channel arrays — switching logic gates (ms-scale gating).
- RAM: Vmem gradient across syncytium — volatile read/write bioelectric memory.
- SSD: Innexin-gated connectivity patterns — non-volatile anatomical memory \
(persists through regeneration indefinitely).

COMPARATIVE ANALYSIS ENGINE — Model Organisms:
- Planarians: decentralized, regenerative anatomical memory.
- Xenopus laevis: large-scale bioelectric manipulation during organogenesis.
- Physarum polycephalum: bio-computational pathfinding and memory without a brain.
- Mammalian organoids: high-fidelity human-analog testing.

MATHEMATICAL TOOLBOX:
Nernst Equation: E_ion = (RT / zF) * ln([Ion]_out / [Ion]_in)
R = 8.314 J/(mol*K), T = temperature in K, z = ion valence, F = 96485 C/mol.
GHK Equation (multi-ion resting potential):
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
- Cable Equation (signal attenuation bounds).
- Conservation of charge, mass, and energy.
- Thermodynamic constraints on any proposed bioelectric process.

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
- ISA Command Mapping: for each step, specify the corresponding ISA \
command (SET BIT, CLEAR BIT, SYNC, CRC CHECK, READ, WRITE).

FALSIFICATION PROTOCOL (MANDATORY):
Every experiment must include a Kill Condition — a specific, measurable \
threshold that REJECTS the hypothesis if reached. Kill Conditions must be \
falsifiable, quantitative, and testable with the hardware specified in the \
execution script.

CLOSED-LOOP EXPERIMENT PROTOCOL (LILA-STYLE):
For every BIM (Biological Information Module), output a Closed-Loop Task:
1. Hypothesis: Based on [Cited Paper].
2. Experiment Design: [Protocol from Wet-Lab Execution Script].
3. Data Collection Plan: [Instrument, units, expected range, sampling rate].
4. Refinement: "If result is X, then BIM parameter Y is valid. If result \
is Z, adjust Graph Connectivity parameter W and re-test."

ERROR CORRECTION & FAULT TOLERANCE:
- Target Morphology = Checksum (pattern validates data integrity).
- Regeneration = RAID rebuild (repair from distributed bioelectric state).
- Colony/tissue redundancy = Replication factor.

THREE-LAYER ARCHITECTURE:
Layer 1 (Knowledge): Immutable primary sources. Cite [1], [2], etc.
Layer 2 (Synthesis): Retrieval-augmented reasoning over structured bioelectric \
variables. Extract: Vmem, EF, Gj, ion channels, perturbations, outcomes.
Layer 3 (Discovery): Comparative analysis, cross-species reasoning, pattern \
detection, hypothesis generation.

SCIENCE-FIRST OUTPUT RULES:
- Tag every sentence: [EVIDENCE], [INFERENCE], [SPECULATION], or [DATA GAP].
- Cite [1], [2], etc. for every factual claim (PMID/DOI where available).
- For every hypothesis: causal chain, missing measurements, falsification experiment.
- Cross-species comparison is mandatory when evidence spans organisms.
- If organism-specific sources are absent, output [DATA GAP] and generate \
targeted collection queries.
- You are a research engine, not a chatbot.

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

CONSTRAINTS:
- No diagnosis or treatment advice.
- Prefer public, de-identified data.
- Prefer "I cannot support that with sources" over inventing details."""

# ======================================================================
# Short, plain-English query mode — the everyday "Query" tab.
# SYSTEM_PROMPT above (and DISCOVERY_TEMPLATE below) are the full Discovery
# Engine used by discover(): heavy, multi-section, hypothesis-and-protocol
# output meant for deep research use. The plain Query button used to share
# that same heavy template, so every quick question came back as a wall of
# academic jargon and wet-lab protocols. This is the same science-first
# rules (no invented numbers, every claim cited, evidence/inference/
# speculation tagged) but told to actually answer in short, plain English
# instead of writing a research proposal.
# ======================================================================
QUERY_SYSTEM_PROMPT = """\
You are Nexus, a research assistant for bioelectricity and biomedical science.
The person asking is NOT a biologist — write for a smart, curious non-scientist.
Anyone technical can click through to the cited source paper to verify.

RULES (non-negotiable):
1. Never invent a number, fact, or citation. Only state what the retrieved
   source passages actually say. If they don't answer the question, say so
   plainly instead of guessing.
2. Every substantive claim goes under one of three headers — EVIDENCE,
   INFERENCE, or SPECULATION:
   - EVIDENCE = directly stated in a cited source.
   - INFERENCE = a reasonable conclusion from the evidence, not stated outright.
   - SPECULATION = plausible but unproven — say so honestly.
3. End each bullet with the source in plain parentheses, e.g.
   "(Mathews & Levin, 2019)" — the app attaches the full citation and a
   link separately, so you don't need DOIs/PMIDs inline.
4. Before any of that, write a SIMPLE VERSION: one or two sentences using a
   plain, everyday analogy — comparing the biology to something like rooms,
   doors, mail, traffic, light switches, plumbing, whatever fits — so someone
   with zero science background gets the gist immediately. Simplify the
   IDEA, not the FACTS: don't say anything the evidence below wouldn't
   support, just say it the simplest possible way. Think "explain it like
   you would to a curious friend who's never taken a biology class," not
   "dumb it down until it's wrong."

STYLE (just as important as the rules above):
- Plain English. Explain any technical term the first time you use it, in a
  few plain words, right in the same sentence, don't assume prior biology
  knowledge.
- Short. One or two sentences per bullet, not paragraphs.
- No headers besides SIMPLE VERSION/EVIDENCE/INFERENCE/SPECULATION, no
  numbered sections, no experiment protocols, no "instruction set" jargon,
  this is a quick answer, not a research proposal.
- No preamble. Don't say "Great question" or restate the question back.
- Never use an em dash (—). Use a comma, period, or "and"/"but" instead."""

SHORT_QUERY_TEMPLATE = """\
Retrieved source passages:

=== SOURCE PASSAGES ===
{context}
========================

Question: {query}

Answer using ONLY what's in the passages above — never invent a fact or
number. Structure your response exactly like this (omit EVIDENCE/INFERENCE/
SPECULATION if you have nothing to put in it, but ALWAYS include SIMPLE
VERSION):

SIMPLE VERSION:
- One or two sentences, plain everyday analogy, zero jargon. This is the
  part a total beginner reads.

EVIDENCE:
- Plain-English bullets stating only what the sources directly say.

INFERENCE:
- Plain-English bullets: reasonable conclusions from the evidence above that
  the sources don't state outright.

SPECULATION:
- Plain-English bullets: plausible ideas that go beyond what the evidence
  actually shows. Say clearly that these are unproven.

If the passages don't answer the question, say so plainly under SIMPLE
VERSION and EVIDENCE instead of manufacturing a technical-sounding non-answer."""

DISCOVERY_TEMPLATE = """\
Retrieved source passages from the bioelectricity and biomedical research corpus:

=== SOURCE PASSAGES ===
{context}
========================

Research query: {query}

Tag EVERY output sentence with [EVIDENCE], [INFERENCE], [SPECULATION], or [DATA GAP].
Execute the discovery loop with ALL of the following mandatory sections:

1. EVIDENCE EXTRACTION
Key findings directly from the sources. Cite each with [1], [2], etc.

2. VARIABLE EXTRACTION
Structured bioelectric variables. Format each as name=value (unit) [source].
Prioritize: Vmem (membrane voltage), EF (electric fields), Gj (gap junctional \
conductance), ion channel types (K+, Na+, Ca2+, Cl-), perturbations, outcomes.
Include organism and cell type context for each variable.
When exact values are absent, calculate E_ion using the Nernst Equation:
  E_ion = (RT / zF) * ln([Ion]_out / [Ion]_in)
Use nearest phylogenetic neighbor concentrations if exact values are unknown.
Label ALL calculated values as [BOUNDED-INFERENCE] — never present as measured data.

3. BIGR INTEGRATION
Map findings to the Bio-Information Genome Runtime layers:
- ROM (DNA): relevant genes, pathways (Wnt, Notum, piwi-1, connexins, innexins)
- RAM (Bioelectricity): Vmem, EF, Gj changes and their causal roles
- Interface (Proteome): ion pumps (H+,K+-ATPase), channels, gap junction proteins
Link these layers explicitly where source data supports it.

4. PATTERN COMPARISON
Compare findings across sources and across species:
Planaria <-> Xenopus <-> Physarum <-> Mammalian organoids.
Note agreements, conflicts, and gaps. Identify conserved bioelectric mechanisms.

5. SUBSTRATE SELECTION MATRIX
When hypotheses involve experimental testing, evaluate model organisms:
| Criteria | Planarian | Xenopus | Physarum | Organoid |
| Ease of Vmem manipulation | ... | ... | ... | ... |
| Data persistence / memory | ... | ... | ... | ... |
| I/O speed (response time) | ... | ... | ... | ... |
| Relevance to query | ... | ... | ... | ... |
Rate each as Low/Medium/High based on sources. If no data, state "No data".

6. HYPOTHESES
Generate testable hypotheses from the patterns. For EACH hypothesis:
- Hypothesis: one falsifiable paragraph with measurable observables \
(mV ranges, time constants, success thresholds). Formal scientific \
language. Describe the proposed mechanism, what information is stored, \
where, how it is read during regeneration.
- This hypothesis is based on:
  * Title — Site/Journal — Author(s) — Year — URL/DOI/PMCID [n]
  * (repeat for all sources used)
- Predicted observables: bullet list, label each MEASURED, PREDICTED, \
SIMULATION-DERIVED, BOUNDED-INFERENCE, or UNKNOWN.

7. EXPERIMENT PROPOSAL
For EACH hypothesis:
A) Simulation: model type, parameters swept, expected outputs, falsification \
criteria. State what parameter is measured (T_hold, BER, Gj, propagation \
speed, attractor count).
B) Wet-lab Phase-0 (cheapest): hardware requirements, reagents/dyes \
(exact markers), timed protocol steps with ISA command mapping (SET_BIT, \
READ_BIT, GATE, AUTH, QUARANTINE, REWRITE), quantification plan (target \
range from GHK), success metric (quantitative pass/fail), kill condition \
(falsifiable rejection threshold), timeline, cost. State parameter \
measured (T_hold, BER, Gj, propagation speed, attractor count).
C) Wet-lab Phase-1 (stronger): same structure as Phase-0.

8. TRANSFER LOGIC
- Planarian→vertebrate mapping rules: gap junctions = innexin (planarian) / \
connexin (vertebrate). State method portability for each experimental step.
- If the hypothesis relies on planarian-specific traits, propose an alternative \
substrate and justify with citations.
- Decision gate: "If X fails, switch to Y substrate."

9. CLOSED-LOOP TASK
For each hypothesis, output a closed-loop refinement protocol:
1. Hypothesis: Based on [Cited Paper].
2. Experiment Design: [Protocol from Experiment Proposal above].
3. Data Collection Plan: [Instrument, units, expected range, sampling rate].
4. Refinement: "If result is X, then BIM parameter Y is valid. If result \
is Z, adjust Graph Connectivity parameter W and re-test."

10. BIOELECTRIC SCHEMATIC
Describe the hypothesized bioelectric circuit:
"[Trigger] -> [Bioelectric change (Vmem/EF/Gj)] -> [Downstream pathway] -> \
[Outcome]"
Label components as [EVIDENCED], [INFERRED], or [SPECULATIVE].

11. BIM SPECIFICATION
For any claimed bioelectric state or "biological bit":
- For EACH measurable parameter (T_hold, E_bit, BER, entropy, capacity):
  Cite measured value OR state "UNKNOWN—needs measurement" + propose \
the minimal measurement to obtain it.
- Map to Hardware Library: CPU (Nav/Kv arrays), RAM (Vmem gradient), \
SSD (Innexin connectivity patterns).

12. GRAPH TOPOLOGY
Model the tissue as a spatial graph:
- Nodes: cell types (neoblasts, differentiated cells)
- Edges: gap junctions (Gj coupling), endogenous EF gradients
- Predict: state propagation speed, stability of bioelectric pattern
- Identify: critical nodes (hub cells), bottleneck edges

13. CROSS-SPECIES NOTES
Where does the evidence transfer across organisms? Where are gaps?

14. UNCERTAINTY & STRATEGIC RECOMMENDATION
Explicit gaps, missing variables, conflicting evidence, what data would resolve them.
"Low Confidence" is NOT a valid final answer. Commit to a strategic recommendation \
with labeled assumptions and a falsification path.

Be precise. No filler. Every claim must trace to a source number. \
Use the Nernst Equation for heuristic baselines when measured Vmem is absent."""


class RAGPipeline:
    """End-to-end RAG implementing the Nexus three-layer research architecture.

    Standard mode: retrieve → assemble context → generate structured response
    Discovery mode (DEFAULT): retrieve → extract variables → compare →
      hypothesize → build bioelectric schematic → propose validation → log
    """

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        n_retrieve: int = 12,
        n_context: int = 8,
    ) -> None:
        self.settings = get_settings()
        self.store = store or VectorStore()
        self.n_retrieve = n_retrieve
        self.n_context = n_context
        self._llm_client = None
        self._provider: str = self.settings.llm_provider.lower()

    def _get_client(self):
        """Lazy-init the LLM client based on the configured provider."""
        if self._llm_client is not None:
            return self._llm_client

        api_key = self.settings.resolved_llm_api_key
        if not api_key:
            if self._provider == "anthropic":
                hint = "Set ANTHROPIC_API_KEY in your environment or .env file."
            else:
                hint = "Set ACHERON_LLM_API_KEY or OPENAI_API_KEY in your environment or .env file."
            raise RuntimeError(
                f"No API key configured for provider '{self._provider}'. {hint}"
            )

        if self._provider == "anthropic":
            import anthropic

            self._llm_client = anthropic.Anthropic(api_key=api_key)
        else:
            from openai import OpenAI

            base_url = self.settings.resolved_llm_base_url
            self._llm_client = OpenAI(api_key=api_key, base_url=base_url)

        return self._llm_client

    # ------------------------------------------------------------------
    # Standard query (Layer 1+2+3)
    # ------------------------------------------------------------------
    def query(
        self,
        question: str,
        filter_source: Optional[str] = None,
        n_results: Optional[int] = None,
    ) -> RAGResponse:
        """Run the full RAG pipeline for a question."""
        n = n_results or self.n_retrieve

        # Layer 2 — Index: retrieve
        results = self.store.search(
            query=question, n_results=n, filter_source=filter_source
        )
        logger.info("Retrieved %d chunks for query", len(results))

        if not results:
            return RAGResponse(
                query=question,
                answer="No relevant sources found in the index. "
                "The Library contains no material matching this query. "
                "Add papers with 'acheron collect' or 'acheron add'.",
                sources=[],
                model_used=self.settings.resolved_llm_model,
            )

        # Select top context passages (source diversity)
        top_results = self._select_context(results)

        # Layer 3 — Compute: generate a short, plain-English response.
        # (Deliberately NOT the heavy Discovery-loop SYSTEM_PROMPT/DISCOVERY_TEMPLATE
        # used below by discover() — that's for deep research, not a quick question.)
        context_str = self._format_context(top_results)
        user_prompt = SHORT_QUERY_TEMPLATE.format(context=context_str, query=question)
        raw_answer = self._generate_with_system(QUERY_SYSTEM_PROMPT, user_prompt, max_tokens=900)

        # Parse structured sections from the response
        plain_summary, evidence, inference, speculation, schematic = (
            self._parse_epistemic_sections(raw_answer)
        )

        return RAGResponse(
            query=question,
            answer=raw_answer,
            sources=top_results,
            model_used=self.settings.resolved_llm_model,
            total_chunks_searched=len(results),
            plain_summary=plain_summary,
            evidence_statements=evidence,
            inference_statements=inference,
            speculation_statements=speculation,
            bioelectric_schematic=schematic,
        )

    # ------------------------------------------------------------------
    # Discovery loop (full Layer 3)
    # ------------------------------------------------------------------
    def discover(
        self,
        question: str,
        filter_source: Optional[str] = None,
        n_results: Optional[int] = None,
        auto_route: bool = True,
    ) -> DiscoveryResult:
        """Execute the full discovery loop.

        Science-First pipeline stages:
        A. Parse query into entities/measurements/constraints
        B. Retrieve from local corpus
        C. Score and filter evidence (organism match, primary data, etc.)
        D. Generate structured output with claim extraction
        E. Synthesize: evidence, data gaps, hypotheses, collection queries

        If auto_route=True (default), queries that are better suited for
        Decision or Tutor mode will be automatically routed to analyze().
        """
        from acheron.models import NexusMode
        from acheron.rag.hypothesis_engine import detect_mode
        from acheron.rag.query_parser import generate_collection_queries, parse_query

        # Smart routing: detect if this query needs a tighter mode
        if auto_route:
            detected_mode = detect_mode(question)
            if detected_mode in (NexusMode.DECISION, NexusMode.TUTOR):
                logger.info(
                    "Auto-routing to %s mode (detected from query)",
                    detected_mode.value,
                )
                # Route to analyze() which handles these modes properly
                result = self.analyze(
                    question=question,
                    mode=detected_mode.value,
                    filter_source=filter_source,
                    n_results=n_results,
                )
                # Convert HypothesisEngineResult to DiscoveryResult for compatibility
                # Extract evidence claims from the graph
                evidence_claims = []
                if result.evidence_graph and result.evidence_graph.claims:
                    for claim in result.evidence_graph.claims:
                        claim_text = claim.subject
                        if claim.predicate:
                            claim_text += f" {claim.predicate}"
                        if claim.object:
                            claim_text += f" {claim.object}"
                        if claim.source_refs:
                            claim_text += f" {' '.join(claim.source_refs)}"
                        evidence_claims.append(claim_text)

                return DiscoveryResult(
                    query=question,
                    evidence=evidence_claims,
                    inference=[],
                    speculation=[],
                    hypotheses=[],
                    sources=result.sources,
                    model_used=result.model_used,
                    total_chunks_searched=result.total_chunks_searched,
                    uncertainty_notes=result.uncertainty_notes,
                    detected_mode=detected_mode.value,
                    # Use the actual LLM response, not just metadata
                    raw_output=result.raw_output,
                )

        n = n_results or self.n_retrieve

        # Stage A — Query understanding
        parsed = parse_query(question)
        logger.info("Query parsed: %s", parsed.summary())

        # Stage B — Retrieve
        results = self.store.search(
            query=question, n_results=n, filter_source=filter_source
        )

        if not results:
            collection_queries = generate_collection_queries(parsed)
            return DiscoveryResult(
                query=question,
                evidence=["No sources retrieved."],
                uncertainty_notes=[
                    "Library contains no material for this query.",
                    "Suggested collection queries:",
                    *[f"  - {q}" for q in collection_queries],
                ],
                model_used=self.settings.resolved_llm_model,
            )

        # Stage C — Score and filter (via _select_context with science-first)
        top_results = self._select_context(
            results, target_organism=parsed.organism_constraint
        )

        # Stage D+E — Discovery compute
        context_str = self._format_context(top_results)
        user_prompt = DISCOVERY_TEMPLATE.format(context=context_str, query=question)
        raw_output = self._generate(user_prompt, max_tokens=3000)

        # Parse the structured discovery output
        return self._parse_discovery_output(
            raw_output=raw_output,
            query=question,
            sources=top_results,
            total_searched=len(results),
        )

    # ------------------------------------------------------------------
    # Retrieve-only (Layer 2 only)
    # ------------------------------------------------------------------
    def retrieve_only(
        self,
        question: str,
        n_results: int = 10,
        filter_source: Optional[str] = None,
    ) -> list[QueryResult]:
        """Retrieve relevant passages without invoking Compute layer."""
        return self.store.search(
            query=question, n_results=n_results, filter_source=filter_source
        )

    # ------------------------------------------------------------------
    # Evidence-Bound Hypothesis Engine (MODE 1/2/3)
    # ------------------------------------------------------------------
    def analyze(
        self,
        question: str,
        mode: Optional[str] = None,
        live: bool = False,
        filter_source: Optional[str] = None,
        n_results: Optional[int] = None,
    ):
        """Run the Evidence-Bound Hypothesis Engine.

        1. Retrieve from local vectorstore.
        2. If evidence is weak and live=True (or auto), fetch from PubMed/bioRxiv/arXiv.
        3. Detect mode (evidence / hypothesis / synthesis) from query or explicit param.
        4. Generate structured output with evidence graph + ranked hypotheses.
        5. Optionally persist high-relevance live papers.

        Returns a HypothesisEngineResult.
        """
        from acheron.models import HypothesisEngineResult, NexusMode
        from acheron.rag.hypothesis_engine import (
            build_engine_result,
            detect_mode,
            get_mode_prompt,
            get_mode_query_template,
        )
        from acheron.rag.live_retrieval import (
            evidence_is_weak,
            live_search,
            persist_high_relevance,
        )

        n = n_results or self.n_retrieve
        detected_mode = detect_mode(question, explicit_mode=mode)
        logger.info("Analyze: mode=%s, live=%s", detected_mode.value, live)

        # Layer 2 — retrieve from local store
        results = self.store.search(
            query=question, n_results=n, filter_source=filter_source
        )
        logger.info("Local retrieval: %d chunks", len(results))

        live_count = 0

        # Live retrieval: trigger if --live flag or evidence is weak
        if live or evidence_is_weak(results):
            logger.info("Triggering live retrieval for query: %s", question)
            try:
                live_results, live_papers, live_chunk_count = live_search(question)
                live_count = live_chunk_count
                # Merge live results with local results
                results.extend(live_results)
                logger.info(
                    "Live retrieval added %d chunks from %d papers",
                    live_chunk_count,
                    len(live_papers),
                )
                # Persist high-relevance papers
                if live_papers:
                    persisted = persist_high_relevance(live_papers)
                    if persisted:
                        logger.info("Persisted %d live papers to Library", persisted)
            except Exception as exc:
                logger.warning("Live retrieval failed: %s", exc)

        if not results:
            return HypothesisEngineResult(
                query=question,
                mode=detected_mode,
                uncertainty_notes=[
                    "No sources retrieved from local index or live search.",
                    "Add papers with 'acheron collect' or try --live flag.",
                ],
                model_used=self.settings.resolved_llm_model,
            )

        # If evidence is still weak after live fetch, auto-enter hypothesis mode
        # (but never override decision mode — the user wants a verdict)
        if detected_mode == NexusMode.EVIDENCE and evidence_is_weak(results):
            detected_mode = NexusMode.HYPOTHESIS
            logger.info("Auto-escalated to hypothesis mode due to weak evidence")

        top_results = self._select_context(results)

        # Layer 3 — Compute with mode-specific prompt
        system_prompt = get_mode_prompt(detected_mode)
        query_template = get_mode_query_template(detected_mode, query=question)
        context_str = self._format_context(top_results)
        user_prompt = query_template.format(context=context_str, query=question)

        raw_output = self._generate_with_system(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=4000,
        )

        return build_engine_result(
            raw_output=raw_output,
            query=question,
            mode=detected_mode,
            sources=top_results,
            total_searched=len(results),
            model_used=self.settings.resolved_llm_model,
            live_sources_fetched=live_count,
        )

    def _generate_with_system(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 2048
    ) -> str:
        """Call the LLM with a custom system prompt (for mode-specific prompts)."""
        model = self.settings.resolved_llm_model

        try:
            client = self._get_client()
        except RuntimeError as exc:
            logger.error("LLM client init failed: %s", exc)
            return (
                f"Error: Compute layer unavailable. Provider '{self._provider}' "
                f"is misconfigured.\n{exc}\n"
                "Retrieval-only mode still works (acheron query -r)."
            )

        try:
            if self._provider == "anthropic":
                # No `temperature` here — newer Claude models (e.g. claude-sonnet-5)
                # reject it as deprecated (400 invalid_request_error). The
                # OpenAI-compatible branch below still accepts it fine.
                response = client.messages.create(
                    model=model,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    max_tokens=max_tokens,
                )
                return response.content[0].text if response.content else ""
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
        except Exception as exc:
            logger.exception(
                "LLM generation failed (Compute layer, provider=%s)", self._provider
            )
            return (
                f"Error: Compute layer unavailable. Provider '{self._provider}' "
                f"is misconfigured.\nRetrieval-only mode still works (acheron query -r). "
                f"Check your API key and provider settings.\nDetail: {exc}"
            )

    # ------------------------------------------------------------------
    # Internal — context selection
    # ------------------------------------------------------------------
    def _select_context(
        self,
        results: list[QueryResult],
        target_organism: str = "",
    ) -> list[QueryResult]:
        """Select the best passages for context, deduplicating by paper.

        When SCIENCE_FIRST_MODE is enabled, uses the evidence scoring
        function (Stage C) instead of raw vector similarity.
        """
        settings = get_settings()
        organism = target_organism or settings.organism_strict

        if settings.science_first_mode:
            from acheron.rag.science_filter import rank_and_filter

            strict = organism != "any"
            scored = rank_and_filter(
                results,
                target_organism=organism if organism != "any" else "planarian",
                top_k=self.n_context,
                strict_organism=strict,
            )
            if scored:
                return [s.result for s in scored]
            # Fallback: if strict filtering returned nothing, use non-strict
            scored = rank_and_filter(
                results,
                target_organism=organism if organism != "any" else "planarian",
                top_k=self.n_context,
                strict_organism=False,
            )
            if scored:
                return [s.result for s in scored]

        # Fallback: original selection logic
        sorted_results = sorted(
            results, key=lambda r: r.relevance_score, reverse=True
        )

        selected: list[QueryResult] = []
        papers_included: dict[str, int] = {}
        max_per_paper = 3

        for r in sorted_results:
            if len(selected) >= self.n_context:
                break
            count = papers_included.get(r.paper_id, 0)
            if count >= max_per_paper:
                continue
            selected.append(r)
            papers_included[r.paper_id] = count + 1

        return selected

    def _format_context(self, results: list[QueryResult]) -> str:
        """Format retrieved passages into a numbered context block.

        Includes PMID/PMCID identifiers and evidence span locations
        for precise citation tracing by the LLM.
        """
        parts = []
        for i, r in enumerate(results, 1):
            author_str = ", ".join(r.authors[:3])
            if len(r.authors) > 3:
                author_str += " et al."
            doi_str = f" DOI: {r.doi}" if r.doi else ""
            pmid_str = f" PMID:{r.pmid}" if r.pmid else ""
            pmcid_str = f" PMCID:{r.pmcid}" if r.pmcid else ""
            header = f"[{i}] {author_str}. \"{r.paper_title}\".{doi_str}{pmid_str}{pmcid_str}"
            section_note = f" (Section: {r.section})" if r.section else ""
            # Include evidence span location if available
            span_note = ""
            if r.source_file and r.xpath:
                span_note = f" [{r.source_file}#{r.xpath}]"
            elif r.source_file and (r.span_start or r.span_end):
                span_note = f" [{r.source_file}:{r.span_start}-{r.span_end}]"
            parts.append(f"{header}{section_note}{span_note}\n{r.text}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Internal — LLM generation
    # ------------------------------------------------------------------
    def _generate(self, user_prompt: str, max_tokens: int = 2048) -> str:
        """Call the LLM (Compute layer). Treats API calls as scarce."""
        model = self.settings.resolved_llm_model

        try:
            client = self._get_client()
        except RuntimeError as exc:
            logger.error("LLM client init failed: %s", exc)
            return (
                f"Error: Compute layer unavailable. Provider '{self._provider}' is misconfigured.\n"
                f"{exc}\n"
                "Retrieval-only mode still works (acheron query -r)."
            )

        try:
            if self._provider == "anthropic":
                # No `temperature` here — see the same note in _generate_with_system.
                response = client.messages.create(
                    model=model,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                    max_tokens=max_tokens,
                )
                return response.content[0].text if response.content else ""
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
        except Exception as exc:
            logger.exception("LLM generation failed (Compute layer, provider=%s)", self._provider)
            return (
                f"Error: Compute layer unavailable. Provider '{self._provider}' is misconfigured.\n"
                f"Retrieval-only mode still works (acheron query -r). "
                f"Check your API key and provider settings.\n"
                f"Detail: {exc}"
            )

    # ------------------------------------------------------------------
    # Internal — parsing structured output
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_epistemic_sections(
        text: str,
    ) -> tuple[str, list[str], list[str], list[str], str]:
        """Parse SIMPLE VERSION / EVIDENCE / INFERENCE / SPECULATION /
        BIOELECTRIC SCHEMATIC from LLM output.

        Returns (plain_summary, evidence, inference, speculation, bioelectric_schematic).
        """
        plain_summary_lines: list[str] = []
        evidence: list[str] = []
        inference: list[str] = []
        speculation: list[str] = []
        schematic_lines: list[str] = []

        current: list[str] | None = None

        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            upper = stripped.upper()
            if any(
                marker in upper
                for marker in [
                    "SIMPLE VERSION", "## SIMPLE", "**SIMPLE",
                ]
            ):
                current = plain_summary_lines
                continue
            if any(
                marker in upper
                for marker in [
                    "EVIDENCE SNAPSHOT", "I. EVIDENCE",
                    "EVIDENCE", "## EVIDENCE", "**EVIDENCE",
                ]
            ):
                if "EXTRACTION" not in upper:
                    current = evidence
                    continue
            if any(
                marker in upper
                for marker in [
                    "DATA GAP", "II. DATA GAP", "II. WHAT WE",
                ]
            ):
                current = None  # captured in raw answer
                continue
            if any(
                marker in upper
                for marker in ["INFERENCE", "## INFERENCE", "**INFERENCE"]
            ):
                current = inference
                continue
            if any(
                marker in upper
                for marker in [
                    "HYPOTHES", "III. HYPOTHES",
                    "SPECULATION", "## SPECULATION", "**SPECULATION",
                ]
            ):
                current = speculation
                continue
            if any(
                marker in upper
                for marker in [
                    "BIOELECTRIC SCHEMATIC", "IV. BIOELECTRIC",
                    "## BIOELECTRIC", "**BIOELECTRIC",
                ]
            ):
                current = schematic_lines
                continue
            if any(
                marker in upper
                for marker in [
                    "CROSS-SPECIES", "V. CROSS",
                    "UNCERTAINTY", "## UNCERTAINTY", "**UNCERTAINTY",
                    "VALIDATION", "## VALIDATION",
                    "MINIMAL WET-LAB", "VI. MINIMAL",
                    "NEXT COLLECTION", "VII. NEXT",
                ]
            ):
                current = None  # captured in raw answer
                continue

            if current is not None and stripped.lstrip("- "):
                current.append(stripped.lstrip("- "))

        schematic = "\n".join(schematic_lines) if schematic_lines else ""
        plain_summary = " ".join(plain_summary_lines) if plain_summary_lines else ""
        return plain_summary, evidence, inference, speculation, schematic

    @staticmethod
    def _parse_discovery_output(
        raw_output: str,
        query: str,
        sources: list[QueryResult],
        total_searched: int,
    ) -> DiscoveryResult:
        """Parse the full discovery loop output into structured fields."""
        evidence: list[str] = []
        inference: list[str] = []
        speculation: list[str] = []
        variables: list[StructuredVariable] = []
        hypotheses: list[Hypothesis] = []
        uncertainty: list[str] = []
        schematic_lines: list[str] = []
        validation_path: list[str] = []
        cross_species: list[str] = []

        current_section = ""
        # Buffer for accumulating multi-line hypothesis blocks
        _hyp_buffer: list[str] = []

        for line in raw_output.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            upper = stripped.upper()

            # Detect section headers
            if any(m in upper for m in [
                "EVIDENCE EXTRACTION", "EVIDENCE EXTRACTED",
                "EVIDENCE —", "1. EVIDENCE", "1) EVIDENCE",
            ]):
                current_section = "evidence"
                continue
            elif any(m in upper for m in ["VARIABLE EXTRACTION", "VARIABLE —", "2. VARIABLE"]):
                current_section = "variables"
                continue
            elif any(m in upper for m in ["BIGR INTEGRATION", "BIGR —", "3. BIGR"]):
                current_section = "bigr"
                continue
            elif any(m in upper for m in ["PATTERN COMPARISON", "PATTERN —", "4. PATTERN"]):
                current_section = "patterns"
                continue
            elif any(
                m in upper
                for m in ["SUBSTRATE SELECTION", "5. SUBSTRATE"]
            ):
                current_section = "substrate"
                continue
            elif any(m in upper for m in ["HYPOTHES", "6. HYPOTHES"]):
                current_section = "hypotheses"
                continue
            elif any(
                m in upper
                for m in [
                    "EXPERIMENT PROPOSAL", "7. EXPERIMENT",
                ]
            ):
                current_section = "validation"
                continue
            elif any(
                m in upper
                for m in ["TRANSFER LOGIC", "8. TRANSFER"]
            ):
                current_section = "cross_species"
                continue
            elif any(
                m in upper
                for m in [
                    "CLOSED-LOOP TASK", "CLOSED LOOP TASK",
                    "9. CLOSED",
                ]
            ):
                current_section = "validation"
                continue
            elif any(
                m in upper
                for m in [
                    "BIOELECTRIC SCHEMATIC",
                    "10. BIOELECTRIC", "9. BIOELECTRIC",
                    "7. BIOELECTRIC",
                    "## BIOELECTRIC",
                ]
            ):
                current_section = "schematic"
                continue
            elif any(
                m in upper
                for m in [
                    "PROTOCOL SPECIFICATION", "8. PROTOCOL",
                    "## PROTOCOL SPEC",
                ]
            ):
                current_section = "protocol"
                continue
            elif any(
                m in upper
                for m in ["VALIDATION PATH", "9. VALIDATION", "VALIDATION STRATEG"]
            ):
                current_section = "validation"
                continue
            elif any(
                m in upper
                for m in [
                    "FAULT TOLERANCE", "10. FAULT",
                    "RAID", "## FAULT TOLERANCE",
                ]
            ):
                current_section = "fault_tolerance"
                continue
            elif any(
                m in upper
                for m in [
                    "BIM QUANTIFICATION", "BIM SPECIFICATION",
                    "11. BIM", "10. BIM",
                    "BIOLOGICAL BIT", "BIOLOGICAL INFORMATION",
                ]
            ):
                current_section = "bim"
                continue
            elif any(
                m in upper
                for m in [
                    "GRAPH TOPOLOGY", "12. GRAPH",
                    "SPATIAL GRAPH",
                ]
            ):
                current_section = "graph"
                continue
            elif any(
                m in upper
                for m in [
                    "CROSS-SPECIES", "CROSS SPECIES", "13. CROSS",
                ]
            ):
                current_section = "cross_species"
                continue
            elif any(
                m in upper
                for m in [
                    "UNCERTAINTY", "14. UNCERTAINTY", "## UNCERTAINTY",
                    "STRATEGIC RECOMMEND",
                ]
            ):
                current_section = "uncertainty"
                continue
            elif any(
                m in upper
                for m in [
                    "MINIMAL WET-LAB", "MINIMAL VIABLE",
                    "DATA GAP", "NEXT COLLECTION",
                ]
            ):
                current_section = "uncertainty"
                continue

            content = stripped.lstrip("- *")
            if not content:
                continue

            if current_section == "evidence":
                evidence.append(content)
            elif current_section == "variables":
                var = _try_parse_variable(content)
                if var:
                    variables.append(var)
                else:
                    evidence.append(content)
            elif current_section == "bigr":
                inference.append(content)
            elif current_section == "patterns":
                inference.append(content)
            elif current_section == "substrate":
                cross_species.append(content)
            elif current_section == "hypotheses":
                # Detect hypothesis boundaries (HYPOTHESIS: or [H1] markers)
                if (
                    stripped.upper().startswith("HYPOTHESIS:")
                    or stripped.upper().startswith("- HYPOTHESIS:")
                    or re.match(r"^Hypothesis:\s*$", stripped)
                    or re.match(r"^\[?H\d+\]?[:\.]", stripped)
                ):
                    # Flush previous buffer
                    if _hyp_buffer:
                        hyp = _try_parse_hypothesis("\n".join(_hyp_buffer))
                        if hyp:
                            hypotheses.append(hyp)
                        else:
                            speculation.extend(_hyp_buffer)
                        _hyp_buffer = []
                _hyp_buffer.append(content)
            elif current_section == "schematic":
                schematic_lines.append(content)
            elif current_section == "protocol":
                validation_path.append(content)
            elif current_section == "validation":
                validation_path.append(content)
            elif current_section == "fault_tolerance":
                inference.append(content)
            elif current_section == "bim":
                inference.append(content)
            elif current_section == "graph":
                inference.append(content)
            elif current_section == "cross_species":
                cross_species.append(content)
            elif current_section == "uncertainty":
                uncertainty.append(content)

        # Flush any remaining hypothesis buffer
        if _hyp_buffer:
            hyp = _try_parse_hypothesis("\n".join(_hyp_buffer))
            if hyp:
                hypotheses.append(hyp)
            else:
                speculation.extend(_hyp_buffer)

        settings = get_settings()
        return DiscoveryResult(
            query=query,
            evidence=evidence,
            inference=inference,
            speculation=speculation,
            variables=variables,
            hypotheses=hypotheses,
            bioelectric_schematic="\n".join(schematic_lines),
            validation_path=validation_path,
            cross_species_notes=cross_species,
            sources=sources,
            model_used=settings.resolved_llm_model,
            total_chunks_searched=total_searched,
            uncertainty_notes=uncertainty,
        )


# ======================================================================
# Parsing helpers
# ======================================================================
def _try_parse_variable(text: str) -> StructuredVariable | None:
    """Attempt to parse a variable from 'name=value (unit) [source]' format."""
    if "=" not in text:
        return None
    try:
        name_part, rest = text.split("=", 1)
        name = name_part.strip()

        source_ref = ""
        if "[" in rest and "]" in rest:
            bracket_start = rest.index("[")
            bracket_end = rest.index("]") + 1
            source_ref = rest[bracket_start:bracket_end].strip()
            rest = rest[:bracket_start].strip()

        unit = ""
        if "(" in rest and ")" in rest:
            paren_start = rest.index("(")
            paren_end = rest.index(")") + 1
            unit = rest[paren_start + 1 : paren_end - 1].strip()
            value = rest[:paren_start].strip()
        else:
            value = rest.strip()

        return StructuredVariable(
            name=name,
            value=value,
            unit=unit,
            source_ref=source_ref,
        )
    except (ValueError, IndexError):
        return None


def _try_parse_hypothesis(text: str) -> Hypothesis | None:
    """Parse a hypothesis from text (single-line or multi-line plain English format).

    Handles both:
    - Old format: single line with inline confidence/impact/assumptions
    - New format: multi-line block with sub-section headers (THE IDEA IN PLAIN
      ENGLISH, WHAT MUST BE TRUE, WHAT IS STILL UNKNOWN, etc.)
    """
    if len(text) < 10:
        return None

    # --- Extract statement (first HYPOTHESIS: line or first line) ---
    statement = text
    lines = text.split("\n")
    for line in lines:
        stripped = line.strip().upper()
        if stripped.startswith("HYPOTHESIS:") or stripped.startswith("- HYPOTHESIS:"):
            statement = line.strip().split(":", 1)[1].strip()
            break
        elif re.match(r"^\[?H\d+\]?[:\.]", line.strip()):
            statement = re.sub(r"^\[?H\d+\]?[:\.\s]*", "", line.strip()).strip()
            break
    else:
        # No explicit marker — use the full text (old single-line format)
        statement = lines[0].strip() if lines else text

    # --- Confidence ---
    confidence = "low"
    lower = text.lower()
    if "high confidence" in lower or "(high)" in lower or "prior confidence: high" in lower:
        confidence = "high"
    elif (
        "medium confidence" in lower
        or "(medium)" in lower
        or "prior confidence: medium" in lower
    ):
        confidence = "medium"

    # --- Refs ---
    refs = []
    for match in re.finditer(r"\[(\d+)\]", text):
        refs.append(match.group(0))

    # --- Parse sub-sections from multi-line plain English format ---
    predicted_impact = ""
    assumptions: list[str] = []
    current_sub = ""

    for line in lines:
        stripped = line.strip()
        upper_line = stripped.upper()

        # Detect sub-section headers
        # --- v3 Discovery Engine format ---
        if "THIS HYPOTHESIS IS BASED ON" in upper_line:
            current_sub = "evidence_for"
            continue
        elif "PREDICTED OBSERVABLE" in upper_line:
            current_sub = "predicts"
            continue
        elif "EXPERIMENT PROPOSAL" in upper_line:
            current_sub = "test"
            continue
        elif upper_line.startswith("SIMULATION:") or "SIMULATION STEP" in upper_line:
            current_sub = "test"
            continue
        elif (
            upper_line.startswith("WET LAB:")
            or "WET-LAB STEP" in upper_line
            or "WET LAB STEP" in upper_line
        ):
            current_sub = "test"
            continue
        elif "PHASE-0" in upper_line or "PHASE 0" in upper_line:
            current_sub = "test"
            continue
        elif "PHASE-1" in upper_line or "PHASE 1" in upper_line:
            current_sub = "test"
            continue
        elif "TRANSFER LOGIC" in upper_line:
            current_sub = "transfer"
            continue
        elif "CLOSED-LOOP TASK" in upper_line or "CLOSED LOOP TASK" in upper_line:
            current_sub = "test"
            continue
        elif "WET-LAB EXECUTION" in upper_line or "WET LAB EXECUTION" in upper_line:
            current_sub = "test"
            continue
        elif "HARDWARE REQUIREMENT" in upper_line:
            current_sub = "test"
            continue
        elif "SENSOR INTEGRATION" in upper_line:
            current_sub = "test"
            continue
        elif "QUANTIFICATION PLAN" in upper_line:
            current_sub = "test"
            continue
        elif "SUCCESS METRIC" in upper_line:
            current_sub = "test"
            continue
        elif "KILL CONDITION" in upper_line:
            current_sub = "test"
            continue
        elif "ISA COMMAND" in upper_line or "SILICON-TO-CARBON" in upper_line:
            current_sub = "test"
            continue
        elif "BOOT PROTOCOL" in upper_line:
            current_sub = "test"
            continue
        elif "DATA COLLECTION PLAN" in upper_line:
            current_sub = "test"
            continue
        elif "REFINEMENT:" in upper_line:
            current_sub = "test"
            continue
        elif "EVIDENCE EXTRACTED" in upper_line:
            current_sub = "evidence_for"
            continue
        # --- v2 format (backward-compatible) ---
        elif "THE IDEA IN PLAIN ENGLISH" in upper_line:
            current_sub = "idea"
            continue
        elif "WHAT MUST BE TRUE" in upper_line:
            current_sub = "assumptions"
            continue
        elif (
            "WHAT WE KNOW" in upper_line
            and "DON'T" not in upper_line
            and "DON\u2019T" not in upper_line
            and "NOT" not in upper_line
        ):
            current_sub = "evidence_for"
            continue
        elif "WHAT WE ALREADY HAVE EVIDENCE" in upper_line:
            current_sub = "evidence_for"
            continue
        elif (
            "WHAT WE DON'T KNOW" in upper_line
            or "WHAT WE DON\u2019T KNOW" in upper_line
        ):
            current_sub = "unknowns"
            continue
        elif "WHAT IS STILL UNKNOWN" in upper_line:
            current_sub = "unknowns"
            continue
        elif "WHAT THIS PREDICTS" in upper_line:
            current_sub = "predicts"
            continue
        elif (
            "PHASE-0 EXPERIMENT" in upper_line
            or "PHASE 0 EXPERIMENT" in upper_line
        ):
            current_sub = "test"
            continue
        elif "FIRST TEST TO VALIDATE" in upper_line or "FIRST TEST:" in upper_line:
            current_sub = "test"
            continue
        elif "WHAT RESULT WOULD PROVE IT WRONG" in upper_line:
            current_sub = "falsif"
            continue
        elif "NEXT 5 QUESTIONS" in upper_line or "NEXT FIVE QUESTIONS" in upper_line:
            current_sub = "questions"
            continue

        # Accumulate into sub-sections
        bullet = stripped.lstrip("- *").strip()
        if not bullet:
            continue

        if current_sub == "idea" or current_sub == "":
            # v3: free-form text after Hypothesis: accumulates as statement
            if predicted_impact:
                predicted_impact += " " + stripped
            else:
                predicted_impact = stripped
        elif current_sub == "assumptions" and bullet:
            assumptions.append(bullet)
        elif current_sub == "evidence_for" and bullet:
            # Evidence items stored as assumptions with tag
            assumptions.append(f"[EVIDENCED] {bullet}")
        elif current_sub == "unknowns" and bullet:
            assumptions.append(f"[UNKNOWN] {bullet}")
        elif current_sub == "predicts" and bullet:
            assumptions.append(f"[PREDICTS] {bullet}")
        elif current_sub == "test" and bullet:
            assumptions.append(f"[EXPERIMENT] {bullet}")
        elif current_sub == "transfer" and bullet:
            assumptions.append(f"[TRANSFER] {bullet}")

    # --- Fallback: old single-line format parsing ---
    if not predicted_impact:
        impact_match = re.search(
            r"predicted impact[:\s]+(.+?)(?:\.|$)", text, re.IGNORECASE
        )
        if impact_match:
            predicted_impact = impact_match.group(1).strip()

    if not assumptions:
        assumption_match = re.search(
            r"assumption[s]?[:\s]+(.+?)(?:\.|$)", text, re.IGNORECASE
        )
        if assumption_match:
            assumptions = [
                a.strip() for a in assumption_match.group(1).split(";") if a.strip()
            ]

    return Hypothesis(
        statement=statement,
        supporting_refs=refs,
        confidence=confidence,
        predicted_impact=predicted_impact,
        assumptions=assumptions,
    )
