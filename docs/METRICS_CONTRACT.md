# Metrics Collector Contract

`sweep.py` integrates with a metrics collector that captures live VAD events
from the OpenAI Realtime API WebSocket stream. In the Talkwise production system 
this is metrics.py in a separate internal repository; for independent implementations, 
any object satisfying the interface below is sufficient.

---

## Required Interface

### `set_vad_config(threshold, silence_duration_ms, prefix_padding_ms)`

Called once per run immediately after the VAD configuration is injected.
Stores the active parameter values so they are included in the snapshot.

| Argument | Type | Description |
|---|---|---|
| `threshold` | `float` | Activation threshold (0–1) |
| `silence_duration_ms` | `int` | Silence duration in milliseconds |
| `prefix_padding_ms` | `int` | Prefix padding in milliseconds |

---

### `set_prompt(prompt: str)`

Called once per run, at the same time as `set_vad_config`, before the
WebSocket session opens. Stores the instruction string sent to OpenAI Realtime
so it is included in the snapshot and in transcript records for post-hoc
LLM-as-a-Judge evaluation.

| Argument | Type | Description |
|---|---|---|
| `prompt` | `str` | Full patient-persona instruction string. Empty string if none. |

The value survives periodic monitoring resets (same preservation rule as
`_vad_config`). Only `_reset_state()` clears it, and only at the start of a
new session.

---

### `reset_sweep_accumulators()`

Called once per run at run start, via `sweep_runner.reset_metrics(metrics_collector)`.
Resets all sweep-level counters so `get_run_snapshot()` reflects only the
current run. Must not reset device info, VAD config, or WebSocket state.

The implementation must maintain **separate sweep accumulators** that are
unaffected by any periodic monitoring reset (e.g. a 30-second cycle). Only
this method may clear them.

---

### `record_transcript_received()`

Called each time a completed transcription event is received
(`conversation.item.input_audio_transcription.completed`).
Used to match VAD triggers to genuine speech turns within the 3-second window.
Takes no arguments — timestamp is recorded internally at call time.

---

### `record_turn(role, text, start_iso, end_iso, silence_before_ms, logprobs)`

Called once per completed conversational turn when `transcript_logging` is
enabled. Appends a structured turn record to the sweep-level transcript
accumulator (`_sweep_transcript`). Only called when the host application has
`transcript_logging=True` in its configuration.

| Argument | Type | Description |
|---|---|---|
| `role` | `str` | `'user'` or `'assistant'` |
| `text` | `str` | Transcribed or generated text for this turn |
| `start_iso` | `str` | ISO datetime string — turn start (VAD speech_started for user; response.created for assistant) |
| `end_iso` | `str` | ISO datetime string — turn end (transcription received / audio transcript done) |
| `silence_before_ms` | `float \| None` | Gap between previous speech_stopped and this speech_started, in ms. `None` for assistant turns and for the first user turn. |
| `logprobs` | `list \| None` | Raw token logprob array from `conversation.item.input_audio_transcription.completed` for user turns; `None` for assistant turns. Requires `include: ['item.input_audio_transcription.logprobs']` in the session configuration. |

The `_sweep_transcript` accumulator is unaffected by periodic monitoring
resets. Only `reset_sweep_accumulators()` may clear it.

---

### `get_run_snapshot() -> dict`

Called once at run end. Must read from sweep accumulators — not from any
counters that may have been reset by a periodic monitoring cycle during the
session. Returns a dict with the following structure:

```python
{
    "vad_config": {
        "threshold": float,
        "silence_duration_ms": int,
        "prefix_padding_ms": int,
    },
    "prompt": str,               # Patient-persona instruction string (empty string if unset)
    "captured_at": str,          # ISO datetime string
    "openai": {
        # Primary dependent variables
        "fpr": float,             # VAD triggers followed by response.done status=cancelled within 1s / total triggers
        "response_latency_ms": float,  # Mean speech_stopped → first audio.delta (ms)

        # Supporting counts
        "vad_triggered": int,     # Total input_audio_buffer.speech_started events
        "vad_false_triggers": int,# Triggers followed by response.done status=cancelled within 1s
        "interruptions": int,     # response.done events with status=cancelled
        "response_count": int,    # Completed model responses

        # Turn duration (ms) — for TDP operationalisation
        "turn_duration_mean_ms": float,
        "turn_duration_count": int,

        # Token usage
        "input_tokens": int,
        "output_tokens": int,

        # Any WebSocket errors during the session
        "errors": list[str],
    },
    "input": {
        "level_dbfs": float,         # Mean input level in dBFS across full session
        "silence_ratio": float,      # Fraction of frames below silence threshold across full session
        "clipping": bool,            # True if clipping detected at any point in session
        "device_name": str,          # Microphone device identifier
        "sample_rate_hardware": int, # Rate the device delivers (e.g. 16000)
        "sample_rate_sent": int,     # Rate sent to OpenAI after resampling (should be 24000)
    },
    "websocket": {
        "latency_ms": float,         # WebSocket round-trip latency (ms), measured via session.update round-trip
        "session_id": str,           # OpenAI session identifier
    },
    "transcript": list[dict],    # Per-turn records; empty list if transcript_logging=False
}
```

Each entry in `transcript` has the following structure:

```python
{
    "role": str,                 # 'user' or 'assistant'
    "text": str,                 # Transcribed or generated text
    "start": str,                # ISO datetime — turn start
    "end": str,                  # ISO datetime — turn end
    "silence_before_ms": float | None,  # ms since previous speech_stopped; None for assistant turns
    "logprobs": list | None,     # Token logprob array for user turns; None for assistant turns
}
```

---

## FPR Operationalisation

`fpr` is computed as:

```
fpr = vad_false_triggers / max(vad_triggered, 1)
```

A trigger is classified as a false positive if the corresponding
`response.done` event carries `status=cancelled` within **1 second** of the
`input_audio_buffer.speech_started` event. `response.done` with
`status=cancelled` is the authoritative signal — the separate
`response.cancelled` event is unreliable in WebSocket mode because the
server may have already completed the response before the cancellation
is processed. Triggers that do not cause a cancellation — because the AI
was already silent — are not counted as false positives.

---

## Turn Latency Operationalisation

`response_latency_ms` is the mean interval between
`input_audio_buffer.speech_stopped` and the first `response.audio.delta`
event of the subsequent model response, measured in milliseconds.
Only completed response pairs (no cancellation) are included in the mean.

---

## Notes

- `get_run_snapshot()` must be safe to call even if the session ended
  prematurely. Return zero-filled fields rather than raising.
- Sweep accumulators must survive any periodic monitoring reset. Only
  `reset_sweep_accumulators()` may clear them.
- `level_dbfs` must be the mean across the full session, not the most
  recent monitoring window. Use a running sum accumulator.
- `silence_ratio` must accumulate across the full session via sweep
  accumulators, not from 30-second monitoring counters. Use
  `_sweep_silence_chunks / _sweep_total_chunks`.
- `clipping` must be True if clipping occurred at any point during the
  session, not just in the last monitoring window.
- `sample_rate_hardware` is the rate the microphone delivers;
  `sample_rate_sent` is the rate transmitted to the API after any
  resampling. Both should be reported — they may differ.
- `sweep.py` calls `get_run_snapshot()` exactly once per run at session end.
- Thread safety is required if the collector is shared across threads.
- `transcript` is always present in the snapshot dict but is an empty list
  when `transcript_logging=False`. Consumers must not assume a non-empty list.
- `prompt` and `transcript` survive periodic monitoring resets. Neither is
  part of the 30-second monitoring payload — they are sweep-level constants
  and accumulators respectively.
- `logprobs` contains the raw per-token logprob array from the
  `conversation.item.input_audio_transcription.completed` event for user turns.
  Requires `include: ['item.input_audio_transcription.logprobs']` in the
  `session.update` payload. `None` for assistant turns — the Realtime API does
  not expose logprobs for generated audio output. Intended as input for ROC-4
  (uncertainty scoring); no downstream consumer currently reads this field.