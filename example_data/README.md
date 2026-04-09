# Example Data

Two example files are provided so you can run the analysis pipeline immediately
without access to a physical device.

**`sweep_vad_example.jsonl`** — 20 synthetic sweep runs (pre-transcript format).
Use with `analyse_sweep.py` and `verify_data.py` for ROC-1 (VAD parameter sweep).

**`sweep_transcript_example.jsonl`** — 3 synthetic sessions in transcript-extended
format (with `prompt` and `transcript` fields). Three VAD configurations are
represented (centre-point, low-threshold corner, high-threshold axial), and one
session contains an out-of-persona assistant turn to exercise the FP detection
path in `judge.py`. Use with `verify_data.py` and `judge.py` for ROC-2
(persona adherence).

---

`sweep_vad_example.jsonl` contains 20 synthetic runs generated with known
parameter-outcome relationships. It is provided so you can run `analyse_sweep.py`
immediately without access to a physical device.

The synthetic FPR is modelled as:

```
FPR ≈ 0.40 − 0.20·x₁ + 0.05·x₂ + noise
```

where `x₁` is coded threshold and `x₂` is coded silence_duration_ms. This
produces a response surface with a known structure that the RSM fitting should
recover.

The 20 records cover all 19 CCD design points (one extra centre-point
replicate), using the Stage 1 parameter levels:

| Parameter | Low | Centre | High |
|-----------|-----|--------|------|
| `threshold` | 0.40 | 0.625 | 0.85 |
| `silence_duration_ms` | 350 ms | 775 ms | 1200 ms |
| `prefix_padding_ms` | 150 ms | 300 ms | 450 ms |

Real sweep data collected during experiments is not included here as it is
device-specific and subject to annotation before publication.

## Record format

`sweep_example.jsonl` uses the **pre-transcript format**: records contain
`vad_config`, `openai`, `websocket`, and `input` inside `snapshot`, but no
`prompt` or `transcript` fields. This is sufficient for ROC-1 (VAD parameter
sweep) and for `analyse_sweep.py`.

Production data collected with `TALKWISE_TRANSCRIPT_LOGGING=true` uses the
**transcript-extended format**, which adds two fields to `snapshot`:

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | `str` | Patient-persona instruction string sent to OpenAI Realtime |
| `transcript` | `list[dict]` | Per-turn records with `role`, `text`, `start_iso`, `end_iso`, `silence_before_ms`, `avg_logprob` |

The transcript-extended format is required for ROC-2 (persona adherence via
`judge.py`). Use `verify_data.py` to check which format a given file uses
before running downstream analysis.