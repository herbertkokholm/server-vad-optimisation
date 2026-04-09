# Paper 2 – Research Sketch

## Working title

**A Receiver Operating Characteristic Framework for Evaluating Decision Boundaries in Multimodal Real-Time Conversational AI**

Alternative:

**A Multi-Threshold Evaluation Framework for Real-Time Conversational AI using LLM-based Event Annotation**

---

# Research motivation

Paper 1 demonstrated that Voice Activity Detection (VAD) can be treated as an empirical optimisation problem. Using Central Composite Design, Response Surface Methodology and Bayesian optimisation, it identified Pareto-optimal operating points balancing False Positive Rate and Turn Latency.

Paper 2 extends this philosophy from a single subsystem to the complete conversational pipeline.

Rather than asking whether an LLM is "good", the paper asks:

> How can every critical decision in a real-time conversational AI system be evaluated and calibrated objectively?

The core hypothesis is that a multimodal conversation consists of a sequence of binary decisions, each having its own decision boundary and therefore its own ROC curve.

---

# Core contribution

The proposed contribution is not another "LLM-as-a-Judge" paper.

Instead it introduces an evaluation methodology where an LLM is used only as an event annotator producing reference labels.

The scientific contribution is a framework that:

* decomposes a real-time conversation into independent binary decision problems
* generates structured event annotations via LLM judge
* evaluates each decision boundary independently
* calibrates thresholds using ROC analysis.

---

# Methodological requirement

For each ROC analysis to qualify as LLM-as-a-Judge evaluation, two independent components must be present:

1. **System-generated continuous score** — the signal produced by the subsystem being calibrated (e.g. VAD threshold, logprob). This is what the ROC sweeps across.
2. **LLM-judge-generated reference label** — a binary ground-truth annotation produced by an LLM reading the conversation transcript. This is what defines correct behaviour.

Both must originate from independent sources. If score and label share the same source the evaluation becomes circular. This requirement governs which ROC analyses are currently implementable and which await further API capabilities.

---

# Overall architecture

```
STT
 ↓
VAD
 ↓
Intent Detection
 ↓
LLM
 ↓
Persona / Dialogue Policy
 ↓
TTS
 ↓
Conversation Transcript + Runtime Metadata
 ↓
LLM Judge
 ↓
Reference Labels
 ↓
ROC Analysis
 ↓
Threshold Optimisation
```

The Judge never optimises the model.

It merely produces reproducible event labels.

---

# Three independent ROC analyses (plus one pending)

ROC-1, ROC-2, and ROC-3 are fully specified and implementable with the current
pipeline. ROC-4 (Uncertainty Handling) is specified but awaits logprob
availability from the Realtime API.

## 1. Turn Taking (VAD)

Positive event:

AI starts speaking at the correct moment.

Continuous score:

* `threshold` — higher threshold is more conservative; trigger events at high threshold are more likely genuine speech. Score direction: `score = threshold` (not `1 − threshold`).

Reference label (LLM judge):

* Judge reads the transcript and annotates each VAD trigger as genuine speech or false alarm. Runtime metadata (`vad_false_triggers`) provides an approximation but does not constitute LLM-as-a-Judge labelling.

Decision:

Did the AI take the turn at the correct moment?

Measures:

* TPR, FPR, AUC
* Optimal silence threshold

---

## 2. Dialogue Policy (Persona Adherence)

Positive event:

AI responds in a manner consistent with the assigned persona.

Continuous score:

* `persona_adherence_score` (0.0–1.0) produced by the LLM judge per assistant turn.

Reference label (LLM judge):

* `should_adhere` / `did_adhere` — binary annotation of whether the AI should have stayed in character and whether it did.

Note on score independence:

Both score and label originate from the LLM judge, capturing different dimensions of the same turn: the continuous score reflects adherence quality, the binary label reflects ground-truth classification. This is defensible but should be acknowledged as a limitation. When intent confidence becomes available via logprobs it can replace the judge-produced score with an independent system signal, resolving the dependency.

Decision:

Did the AI maintain its assigned persona throughout the turn?

Measures:

* TPR, FPR, AUC
* Optimal adherence threshold for flagging character breaks

---

## 3. Conversation Flow Management

Positive event:

A conversational redirect or topic change was needed at this turn (e.g. the user posed an off-topic clinical question the patient should not engage with; the current topic was exhausted).

Continuous score:

* `flow_score` (0.0–1.0) produced by the LLM judge per assistant turn.

Reference label (LLM judge):

* `did_transition_appropriately` — binary annotation of whether the AI handled the conversational moment correctly (redirected when needed, continued when appropriate).

Note on y_true choice:

`flow_score` measures quality of *execution*, not whether a transition was contextually required. It therefore correlates with `did_transition_appropriately`, not with `should_transition`. Using `should_transition` as y_true inverts the curve (AUC < 0.5). The ROC asks: "Can flow_score serve as a threshold for detecting flow management failures?" — an actionable question for runtime flagging.

Note on score independence:

Same structural note as ROC-2: score and label both originate from the judge, capturing different dimensions of the same turn.

Decision:

Did the AI redirect or continue the conversation appropriately?

Measures:

* TPR, FPR, AUC
* Optimal flow threshold for flagging mismanaged transitions

Implementation:

Both ROC-2 and ROC-3 judgments are captured in a single API call per turn via the extended `judge.py` tool schema. No additional API cost relative to ROC-2 alone.

---

## 4. Uncertainty Handling (pending)

Positive event:

AI activates fallback or abstention behaviour when appropriate.

Continuous score:

* Average log probability per output token — higher values indicate higher model confidence.

Reference label (LLM judge):

* Judge annotates whether the AI should have expressed uncertainty or abstained given the question posed.

Status:

The OpenAI Realtime API does not currently expose per-token log probabilities. This ROC analysis is fully specified but pending API availability. Once logprobs are enabled, `roc_analysis.py` will require a new `build_uncertainty_roc_data` function and a corresponding `plot_roc4`; `judge.py` will require an additional `should_abstain` / `did_abstain` field in the tool schema. No structural changes to the pipeline are anticipated.

Decision:

Should the AI answer or abstain?

Measures:

* ROC, AUC
* Optimal logprob cutoff for uncertainty flagging

---

# Independence assumption

ROC curves are treated as statistically independent because each is associated with a distinct subsystem and a distinct decision boundary. However, causal dependence exists downstream: VAD errors propagate to dialogue policy, and dialogue policy errors may affect perceived uncertainty. This structural dependence is acknowledged as a limitation. Per-subsystem analysis remains valid as a diagnostic instrument provided results are not interpreted as causally isolated.

---

# Event taxonomy

Each event is represented as:

```json
{
  "timestamp": "...",
  "event": "...",
  "classification": "TP|FP|TN|FN",
  "primary_cause": "VAD | DialoguePolicy | Uncertainty",
  "score": 0.83,
  "reasoning": "..."
}
```

Every event belongs to exactly one subsystem.

---

# Study design

## Prompt

A single fixed patient persona is used across all sessions. This ensures that LLM judge scores are calibrated on a consistent reference, making scores directly comparable across sessions. Two to three prompts may be used if generalisation is a secondary objective, in which case ROC analyses are stratified per prompt.

## Session structure

```
[0 – 8 min]   Simulation in-character
              AI plays patient. All assistant turns have should_adhere = True.
              Produces TP and FN events for ROC-2.

[8 – 10 min]  Structured debrief
              Facilitator explicitly closes the simulation.
              All subsequent assistant turns have should_adhere = False.
              Produces TN and FP events for ROC-2.
              Without this phase ROC-2 has only one label class and is not computable.

[Throughout]  VAD metadata collected for ROC-1.
              Logprob scores collected for ROC-3 when available.
```

## Sample size

Driven primarily by ROC-1, which requires coverage of the VAD parameter space.

| Design points (threshold × silence_duration) | Repetitions per point | Total sessions | Estimated false-trigger events |
|---|---|---|---|
| 10 | 3 | 30 | ~90 — marginal |
| 15 | 4 | 60 | ~180 — adequate |
| 20 | 5 | 100 | ~300 — robust |

**Recommended minimum: N = 50 sessions** covering 10–15 VAD configurations with 3–5 repetitions each. This yields 400–500 judge evaluations for ROC-2 and sufficient trigger events for a stable ROC-1 curve (SE ≤ 0.05).

## Participants

Medical students or healthcare professionals with no prior exposure to the specific simulation scenario. Between-subjects design preferred to avoid learning and fatigue effects. If within-subjects, counterbalancing of VAD configurations is required.

## Judge reproducibility

LLM judge runs at temperature = 0. Inter-run agreement (Cohen's κ) is reported by running a held-out subset of sessions twice. This establishes the reliability of reference labels before ROC analyses are computed.

---

# Experimental pipeline

```
Conversation
 ↓
Runtime metadata + transcript
 ↓
LLM Judge  →  Reference labels (per turn, per subsystem)
 ↓
scikit-learn
 ↓
ROC curves (one per subsystem)
 ↓
Threshold optimisation
```

---

# Research questions

RQ1

Can real-time conversational behaviour be decomposed into independent binary decision problems?

RQ2

Can LLM-generated event annotations produce reproducible reference labels suitable for ROC analysis?

RQ3

Which subsystem contributes most to degraded conversational quality?

RQ4

Can independent threshold calibration improve overall dialogue quality without modifying the underlying LLM?

---

# Relationship to Paper 1

Paper 1:

One decision boundary. One ROC. One optimisation problem.

Paper 2:

Three decision boundaries now (VAD, Persona, Flow). Four when logprobs are available (+Uncertainty). One unified evaluation framework.

The methodology scales naturally from subsystem optimisation to end-to-end conversational optimisation.

---

# Future Paper 3

Adaptive online threshold optimisation.

Runtime Bayesian optimisation.

Continuous learning.

Automatic threshold adjustment based on observed interaction quality.

---

# Suggested recent literature

## LLM-as-a-Judge

* Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.
* Gu et al. (2024). A Survey on LLM-as-a-Judge.
* Huang et al. (2025). An Empirical Study of LLM-as-a-Judge Reliability.

## Voice Activity Detection

* Skantze (2021). Turn-taking in Conversational Systems and Human-Robot Interaction.
* OpenAI Realtime VAD Documentation (2025).
* Jiang et al. (2025). Small-Footprint Acoustic Echo Cancellation for Full Duplex Speech Interaction.

## Intent Detection

* Zhang et al. (2024). Large Language Models for Intent Detection: A Survey.
* Qin et al. (2023). Prompting Large Language Models for Intent Classification.

## Uncertainty and Hallucination

* Kuhn et al. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in LLMs.
* Lin et al. (2022). Teaching Models to Express Their Uncertainty.
* Xiong et al. (2024). Hallucination Detection and Uncertainty Calibration in Large Language Models.

## ROC and Decision Threshold Optimisation

* Fawcett (2006). An Introduction to ROC Analysis.
* Saito & Rehmsmeier (2015). The Precision-Recall Plot is More Informative than ROC for Imbalanced Datasets.
* Recent applications of ROC calibration in LLM evaluation (2024–2025).
