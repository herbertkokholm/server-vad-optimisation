# Sources

Core implementation of the three-stage VAD optimisation methodology.

| File | Stage | Description |
|------|-------|-------------|
| [`sweep.py`](sweep.py) | 1 | CCD sweep runner — iterates through 76-run design matrix and writes JSONL output per session. |
| [`analyse_sweep.py`](analyse_sweep.py) | 2 | Post-hoc analysis — fits RSM models, computes Pareto front, produces figures and `pareto_model.json`. |
| [`adaptive_tuner.py`](adaptive_tuner.py) | 3 | Session-level adaptive tuner — observes live VAD metrics and emits `session.update` events toward the Pareto-optimal configuration. |
| [`judge.py`](judge.py) | — | LLM-as-a-Judge — scores each AI turn in a transcript for persona adherence (0.0–1.0) and classifies turns as TP/FP/TN/FN. Input: sweep or transcript JSONL. Output: annotated JSONL for ROC analysis. |
| [`roc_analysis.py`](roc_analysis.py) | — | ROC curve generation — produces `fig1_roc_vad.png` (ROC-1: VAD trigger classification across threshold settings) and `fig2_roc_persona.png` (ROC-2: persona_adherence_score threshold sweep). Input: sweep JSONL and/or `judge.py` output. |
| [`verify_data.py`](verify_data.py) | — | Pre-flight data check — validates that JSONL files contain all fields required for ROC-1 (VAD sweep) and ROC-2 (persona adherence). Exit 0 = ready. |
| [`pareto_model.json`](pareto_model.json) | — | Empirical Pareto front from the Stage 1 CCD sweep (*N* = 76). Loaded by `adaptive_tuner.py`. |
| [`requirements.txt`](requirements.txt) | — | Python dependencies. |

See the repository root [`README.md`](../README.md) for usage and integration details.
