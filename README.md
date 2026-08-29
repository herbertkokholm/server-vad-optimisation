# server-vad-optimisation

**An empirical framework for optimising Voice Activity Detection parameters in
LLM-driven conversational systems, extended with an LLM-as-a-Judge phase for
intent-detection analysis.**

This repository has grown across two research phases:

- **Phase 1 — VAD parameter optimisation** (complete): a CCD sweep runner,
  RSM analysis pipeline, Pareto-front selection, and an adaptive tuning
  algorithm. This phase is the subject of the published paper below.
- **Phase 2 — LLM-as-a-Judge** (in progress): `judge.py` scores transcripts
  for persona adherence and classifies AI turns as TP/FP/TN/FN to support
  intent-detection ROC analysis. Data collection is pending; no paper exists
  yet for this phase.

---

### Phase 1 paper

This repository contains the open-source implementation accompanying the paper:

> *Voice Activity Detection for Conversational AI in Clinical Simulation-Based
> Learning: An Empirical Optimisation Framework*
> — submitted to Computer Speech & Language (under review)

A preprint is available in [`paper/`](papers/paper1/VAD_Optimisation_preprint.pdf).

The paper covers **Phase 1 only** (sweep, RSM, Pareto front, adaptive tuner).
Phase 2 (LLM-as-a-Judge) and its accompanying publication are pending.

---

## Background

### VAD parameter optimisation

Server-side VAD in the OpenAI Realtime API is controlled by three parameters:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `threshold` | 0.5 | Speech onset sensitivity |
| `silence_duration_ms` | 500 | Turn-end detection latency |
| `prefix_padding_ms` | 300 | Pre-speech context buffer |

No empirical framework exists for optimising these jointly in domain-specific
conversational contexts. Misconfigured VAD causes backchannel vocalisations
(`hm`, `mhm`, brief pauses) to be misidentified as new speaker turns, disrupting
conversational naturalness and ecological validity in clinical simulation.

This repository provides:

1. **`sweep.py`** — a Central Composite Design (CCD) sweep runner that iterates
   through a 76-run design matrix (19 design points × 4 repetitions), injects
   VAD parameters into each session, and writes JSONL output per run.

2. **`analyse_sweep.py`** — a post-hoc analysis script that loads JSONL files,
   fits second-order RSM models, computes the Pareto front in FPR–TL objective
   space, produces publication-quality PNG figures, and writes `pareto_model.json`.

3. **`adaptive_tuner.py`** — a session-level adaptive tuning algorithm that
   observes live VAD metrics and emits `session.update` events toward the
   Pareto-optimal configuration. Loads `pareto_model.json` produced by
   `analyse_sweep.py`. **Implemented but not yet empirically validated** —
   see [Adaptive Tuner](#adaptive-tuner) below.

4. **`verify_data.py`** — a pre-flight data check that validates JSONL files
   contain all fields required for downstream ROC analysis. Exit 0 = ready.

### LLM-as-a-Judge

*(Phase 2 — in progress, data collection pending)*

`judge.py` reads transcript-extended sweep JSONL (requires `transcript_logging=True`
and a populated `prompt` field) and, for each AI (assistant) turn, queries
`claude-sonnet-4-6` to assess whether the AI correctly maintained the patient
persona defined in the session's persona instruction. The persona instruction is
read dynamically from the JSONL record, making the pipeline case- and
language-agnostic.

The judge produces the following per-turn parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `should_adhere` | bool | True if the AI should respond in-character as the patient in this context |
| `did_adhere` | bool | True if the AI's actual response stayed in-character |
| `persona_adherence_score` | float 0.0–1.0 | Continuous quality score (1.0 = perfect patient response) |
| `label` | TP \| FP \| TN \| FN | Classification derived from `should_adhere` × `did_adhere` |
| `reason` | str | One-sentence evidence-based explanation |

A `session_adherence_score` (mean of all turn scores) is also written per session.
These turn-level parameters feed into ROC-2 in `roc_analysis.py`, where
`persona_adherence_score` is swept as the discriminator threshold and `should_adhere`
provides the binary ground truth.

A first skecth for a study is listed here: [`paper/`](papers/paper2/paper2_sketch.md).

---

## Reference Hardware Platform

The study data was collected using [herbertkokholm/talkwise-pi](https://github.com/herbertkokholm/talkwise-pi)
— an embedded audio terminal running on Raspberry Pi, built for OSCE clinical
simulation. The production sweep runner (`sweep.py`) and metrics collector
(`metrics.py`) in that repository are the implementations from which this
standalone version is derived.

---

## Quick Start

```bash
git clone https://github.com/herbertkokholm/server-vad-optimisation
cd server-vad-optimisation
pip3 install -r requirements.txt

# Run analysis on the included example data
python3 analyse_sweep.py --input example_data/ --output figures/
```

This produces six figures, a `summary.txt`, and a `pareto_model.json` in `figures/`.

---

## Running a Sweep

`sweep.py` is designed to integrate with any Python-based WebSocket client
targeting the OpenAI Realtime API. The interface is two calls:

```python
from sweep import sweep_runner

# At session start — returns VAD config dict for this run
vad = sweep_runner.next_run(session_id)
# vad = {"threshold": 0.70, "silence_duration_ms": 800, "prefix_padding_ms": 300}

# At session end — saves snapshot and advances design matrix
sweep_runner.end_run(snapshot)
```

Enable sweep mode by setting `SWEEP_ACTIVE = True` in `sweep.py`. When `False`
(the default), `next_run()` returns default VAD values and `end_run()` is a
no-op — existing production behaviour is unchanged.

### Snapshot format

The outer record structure is always the same. The content of `snapshot` depends
on which flags are enabled in the host application:

**Base format** *(Phase 1)* — `sweep_active=True`, `transcript_logging=False`

```json
{
  "run_index": 13, "run_number": 14, "total_runs": 76,
  "design_point_index": 13, "rep": 1, "session_id": 135,
  "started_at": "...", "ended_at": "...", "duration_s": 181.7,
  "annotation": {"valid": null, "warmup": false, "speaker": 1},
  "snapshot": {
    "vad_config": {"threshold": 0.625, "silence_duration_ms": 775, "prefix_padding_ms": 300},
    "prompt": "",
    "captured_at": "...",
    "input": {...},
    "openai": {"fpr": 0.2, "response_latency_ms": 754.9, ...},
    "websocket": {"latency_ms": 205.5, "session_id": "..."},
    "transcript": []
  }
}
```

**Transcript-extended format** *(Phase 2 — data collection pending)* — `sweep_active=True`, `transcript_logging=True`

`prompt` is populated and `transcript` contains per-turn records. This format
is required for Phase 2: the collected JSONL is what `judge.py` reads to score
persona adherence and build the intent-detection ROC dataset.

```json
{
  "snapshot": {
    "vad_config": {...},
    "prompt": "Du er Mette Hansen, 54 år...",
    "captured_at": "...",
    "input": {...},
    "openai": {...},
    "websocket": {...},
    "transcript": [
      {"role": "user", "text": "...", "start_iso": "...", "end_iso": "...",
       "silence_before_ms": 1240.0, "avg_logprob": null},
      {"role": "assistant", "text": "...", "start_iso": "...", "end_iso": "...",
       "silence_before_ms": null, "avg_logprob": null}
    ]
  }
}
```

See [`example_data/sweep_vad_example.jsonl`](example_data/sweep_vad_example.jsonl)
for 20 base-format records, and
[`example_data/sweep_transcript_example.jsonl`](example_data/sweep_transcript_example.jsonl)
for 3 transcript-extended records including one out-of-persona assistant turn.

### State persistence

The sweep runner saves its position in the design matrix to
`sweep_runs/sweep_state.json` after each run. If the process is interrupted,
it resumes from where it left off on the next start.

---

## Analysis

### VAD sweep — `analyse_sweep.py`

```bash
python3 sources/analyse_sweep.py --input sweep_runs/ --output figures/

# Filter options
python3 sources/analyse_sweep.py --input sweep_runs/ --output figures/ --filter valid
python3 sources/analyse_sweep.py --input sweep_runs/ --output figures/ --filter annotated
```

#### Filter modes

| Mode | Behaviour |
|------|-----------|
| `all` | Include all runs regardless of `annotation.valid` |
| `annotated` | Include only runs where `annotation.valid` is not `None` |
| `valid` | Include only runs where `annotation.valid == True` |

Additional thresholds (minimum session duration, FPR outlier cap) are
configurable in the `CONFIG` dict at the top of `analyse_sweep.py`.

#### Output figures

| File | Content |
|------|---------|
| `fig1_response_surfaces.png` | RSM contour plots for FPR and TL across threshold × silence_duration_ms (prefix_padding fixed at centre point) |
| `fig2_pareto_front.png` | Pareto front in FPR–TL objective space |
| `fig3_rsm_coefficients.png` | RSM coefficients with significance markers (Bonferroni-corrected, p < 0.005) |
| `fig4_marginal_effects.png` | Marginal effects of each parameter on FPR and TL |
| `summary.txt` | Descriptive statistics, R², significant terms, Pareto configurations |
| `pareto_model.json` | Pareto-optimal configurations for use by `adaptive_tuner.py` |

All figures are 300 DPI, Times New Roman, suitable for direct LaTeX inclusion.

Response surfaces are visualised as 2D slices with prefix_padding_ms fixed at its centre-point value.

### LLM-as-a-Judge — `judge.py`

Requires `ANTHROPIC_API_KEY` and transcript-extended JSONL (collected with
`transcript_logging=True`).

```bash
export ANTHROPIC_API_KEY=<key>
python3 sources/judge.py --input sweep_runs/ --output judged.jsonl
```

`judge.py` iterates over every assistant turn in each session and calls
`claude-sonnet-4-6` (temperature 0.0) to classify the turn. The annotated
output is a JSONL file with a `judge` key added to each record; sessions
without a transcript or persona prompt are passed through unchanged.

### ROC curves — `roc_analysis.py`

```bash
# Both curves in one run
python3 sources/roc_analysis.py \
    --sweep  sweep_runs/     \
    --judged judged.jsonl    \
    --output figures/

# ROC-1 only (no judged file needed)
python3 sources/roc_analysis.py --sweep sweep_runs/ --output figures/

# ROC-2 only (no sweep file needed)
python3 sources/roc_analysis.py --judged judged.jsonl --output figures/
```

#### Output figures

| File | Content |
|------|---------|
| `fig1_roc_vad.png` | ROC-1: VAD trigger classification — TPR vs FPR as a function of the threshold parameter across all sweep sessions |
| `fig2_roc_persona.png` | ROC-2: persona adherence score as discriminator — TPR vs FPR sweeping `persona_adherence_score` as the classification threshold |

Both figures are 300 DPI, Times New Roman, with AUC reported in the legend.

---

## Annotating Results

Each JSONL record contains an `annotation` field:

```json
"annotation": {
  "valid": null,
  "warmup": false,
  "speaker": 1
}
```

After running sessions, open the JSONL file and set:

- `valid`: `true` if the run followed the protocol, `false` if it should be excluded
- `warmup`: `true` to exclude warm-up runs from analysis
- `speaker`: integer speaker identifier (1, 2, ...); used as a covariate in RSM to control for between-speaker variance

Use `--filter valid` in `analyse_sweep.py` to exclude invalidated runs.

---

## CCD Design

The sweep covers a face-centred Central Composite Design with three factors:

| Factor | Low (−1) | Centre (0) | High (+1) |
|--------|----------|-----------|-----------|
| `threshold` | 0.40 | 0.625 | 0.85 |
| `silence_duration_ms` | 350 ms | 775 ms | 1200 ms |
| `prefix_padding_ms` | 150 ms | 300 ms | 450 ms |

19 design points × 4 repetitions = 76 total runs at approximately 180 seconds
each. A complete sweep takes roughly one working day.

---

## Extending to Other Domains

The sweep runner and analysis pipeline are domain-agnostic. To use with a
different conversational system:

1. Implement the two-call interface (`next_run` / `end_run`) in your session
   lifecycle.
2. Ensure your snapshot dict includes `snapshot.vad_config`, `snapshot.openai.fpr`,
   and `snapshot.openai.response_latency_ms`. For intent-detection ROC analysis
   via `judge.py`, also populate `snapshot.prompt` and `snapshot.transcript`
   (requires `transcript_logging=True` and calls to `set_prompt()` / `record_turn()`).
3. Run `verify_data.py` to confirm the collected files are complete before analysis.
4. Run `analyse_sweep.py` on the collected JSONL files.

The paper discusses cross-domain validation (OSCE clinical simulation →
dementia care dialogue) and a planned `semantic_vad` comparative study.

---

## Adaptive Tuner

`adaptive_tuner.py` implements a session-level adaptive tuning algorithm. It
loads the Pareto front from `pareto_model.json` (produced by `analyse_sweep.py`)
and emits `session.update` events when a better VAD configuration exists on the
Pareto front.

```python
from adaptive_tuner import AdaptiveVADTuner

# Load Pareto model produced by analyse_sweep.py
tuner = AdaptiveVADTuner.from_pareto_file("figures/pareto_model.json", omega=0.7)

# In your WebSocket receive loop:
await tuner.update_if_needed(ws, metrics_collector)
```

The `omega` parameter controls the FPR/TL trade-off: `1.0` minimises FPR
exclusively, `0.0` minimises turn latency exclusively.

**Status:** The algorithm is fully designed and implemented — interface, scoring
function, normalised parameter distance, and `session.update` emission are
complete. `pareto_model.json` is committed to the repository and contains the
empirical Pareto front from the Stage 1 CCD sweep ($N = 76$).
`tuner.is_ready` returns `True` when the model is loaded with scored points.
Empirical validation is pending and planned using a leave-one-domain-out protocol
(OSCE → dementia care dialogue) with multiple naive speakers.

---

## Metrics Collector Contract

`sweep.py` integrates with a metrics collector that captures live VAD events
from the OpenAI Realtime API WebSocket stream. In the Talkwise production system this is
`metrics.py` in [herbertkokholm/talkwise-pi](https://github.com/herbertkokholm/talkwise-pi) —
the embedded audio terminal platform that collected the study data.

The required interface is documented in [`docs/METRICS_CONTRACT.md`](docs/METRICS_CONTRACT.md).
Any object satisfying that contract can be used as a drop-in replacement. The contract
includes five methods:

| Method | Purpose |
|--------|---------|
| `set_vad_config(threshold, silence_ms, prefix_ms)` | Store active VAD parameters for the run |
| `set_prompt(prompt)` | Store persona instruction string for transcript records |
| `reset_sweep_accumulators()` | Clear per-run counters at session start |
| `record_transcript_received()` | Mark a completed user transcription event |
| `record_turn(role, text, start_iso, end_iso, silence_before_ms, avg_logprob)` | Append a turn to the session transcript |

`get_run_snapshot()` returns all accumulated data including `vad_config`, `prompt`,
and `transcript` for downstream use by `judge.py` and `analyse_sweep.py`.

---

## Repository Structure

```
server-vad-optimisation/
├── sources/
│   ├── sweep.py              # CCD sweep runner
│   ├── analyse_sweep.py      # RSM fitting, figure generation, pareto_model.json output
│   ├── adaptive_tuner.py     # Adaptive tuning algorithm (implemented, validation pending)
│   ├── judge.py              # LLM-as-a-Judge — persona adherence scoring (in progress)
│   ├── verify_data.py        # Pre-flight ROC data check
│   ├── pareto_model.json     # Pareto-optimal configurations (output of analyse_sweep.py)
│   └── requirements.txt
├── docs/
│   └── METRICS_CONTRACT.md   # Metrics collector interface specification
├── example_data/
│   ├── sweep_vad_example.jsonl        # 20 synthetic CCD runs (pre-transcript format, ROC-1)
│   ├── sweep_transcript_example.jsonl # 3 synthetic sessions with transcript (ROC-2)
│   └── README.md
├── paper/
│   └── VAD_Optimisation_preprint.pdf
├── LICENSE
├── CITATION.cff
└── .gitignore
```

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{kokholm2026vad,
  title   = {Voice Activity Detection for Conversational AI in Clinical
             Simulation-Based Learning: An Empirical Optimisation Framework},
  author  = {Kokholm, Thomas Herbert},
  journal = {Computer Speech & Language},
  year    = {2026},
  doi     = {10.2139/ssrn.6739487},
  note    = {Under review}
}
```

Or use the GitHub "Cite this repository" button which reads `CITATION.cff`.

---

## License

MIT — see [`LICENSE`](LICENSE).

## Contributing

Cross-domain replications are the most valuable form of contribution. If you run a sweep in a new domain, please open a pull request or issue — see [`CONTRIBUTING.md`](CONTRIBUTING.md) for what to include.