"""
VAD Parameter Sweep Runner for Talkwise.

Manages a Central Composite Design sweep over three VAD parameters:
  - threshold
  - silence_duration_ms
  - prefix_padding_ms

Usage in main.py:

    from sweep import sweep_runner

    # At session start — returns VAD config dict (defaults if sweep inactive)
    vad = sweep_runner.next_run(session_id)

    # At session end — records snapshot and advances design matrix
    sweep_runner.end_run(metrics_collector.get_run_snapshot())

Results are written to sweep_runs/sweep_<timestamp>.jsonl, one line per run.
Sweep state (current position in design matrix) persists across restarts
via sweep_runs/sweep_state.json.

Set SWEEP_ACTIVE = True to enable. When False, next_run() returns DEFAULT_VAD
and end_run() is a no-op — main.py behaviour is unchanged.
"""

import json
import logging
import threading
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Optional

logger = logging.getLogger("talkwise.sweep")

# ── Configuration ─────────────────────────────────────────────────────────────

SWEEP_ACTIVE = True

# Output directory (relative to runtime working directory)
SWEEP_DIR = Path("sweep_runs")

# Default VAD config returned when sweep is inactive
DEFAULT_VAD = {
    "threshold": 0.5,
    "silence_duration_ms": 500,
    "prefix_padding_ms": 300,
}

# Repetitions per design point (5 gives within-cell variance estimates)
REPS_PER_POINT = 5

# ── CCD Design Matrix ─────────────────────────────────────────────────────────
#
# Face-centred Central Composite Design for three factors:
#   x1 = threshold          coded: -1 = 0.40,  0 = 0.625,  +1 = 0.85
#   x2 = silence_duration_ms coded: -1 = 350,   0 = 775,    +1 = 1200
#   x3 = prefix_padding_ms  coded: -1 = 150,   0 = 300,    +1 = 450
#
# Design points:
#   8  factorial corners  (±1, ±1, ±1)
#   6  axial / star       (±1,  0,  0) etc. — face-centred so α = 1
#   5  centre replicates  ( 0,  0,  0)
#   Total: 19 distinct + 1 replicate centre = 20 points × REPS_PER_POINT runs


def _build_ccd() -> list[dict]:
    """Build the full CCD design matrix as a list of VAD config dicts."""

    # Decode coded levels to natural units
    def decode(x1, x2, x3) -> dict:
        threshold = round(0.625 + x1 * 0.225, 4)  # centre=0.625, range 0.40–0.85
        silence_ms = int(775 + x2 * 425)  # centre=775,   range 350–1200
        prefix_ms = int(300 + x3 * 150)  # centre=300,   range 150–450
        return {
            "threshold": threshold,
            "silence_duration_ms": silence_ms,
            "prefix_padding_ms": prefix_ms,
        }

    points = []

    # 8 factorial corners
    for x1, x2, x3 in product([-1, 1], repeat=3):
        points.append(decode(x1, x2, x3))

    # 6 axial points (face-centred: α = 1)
    for axis, val in [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)]:
        coded = [0, 0, 0]
        coded[axis] = val
        points.append(decode(*coded))

    # 5 centre replicates
    for _ in range(5):
        points.append(decode(0, 0, 0))

    # Expand by REPS_PER_POINT (interleaved so each point appears in order)
    expanded = []
    for rep in range(REPS_PER_POINT):
        for i, point in enumerate(points):
            expanded.append(
                {
                    "design_point_index": i,
                    "rep": rep + 1,
                    **point,
                }
            )

    return expanded


# Precompute at import time
CCD_MATRIX = _build_ccd()
TOTAL_RUNS = len(CCD_MATRIX)


# ── State persistence ─────────────────────────────────────────────────────────


class SweepState:
    """Persists sweep position across restarts."""

    def __init__(self, state_path: Path):
        self._path = state_path
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load sweep state: {e}. Starting fresh.")
        return {"run_index": 0, "started_at": datetime.now().isoformat()}

    def _save(self):
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    @property
    def run_index(self) -> int:
        with self._lock:
            return self._data["run_index"]

    def advance(self):
        with self._lock:
            self._data["run_index"] += 1
            self._save()

    def reset(self):
        with self._lock:
            self._data = {"run_index": 0, "started_at": datetime.now().isoformat()}
            self._save()
            logger.info("Sweep state reset — starting from run 1")


# ── Sweep Runner ──────────────────────────────────────────────────────────────


class SweepRunner:
    """
    Main sweep controller. Singleton — import sweep_runner.

    main.py interface:
        vad = sweep_runner.next_run(session_id)   # call before configure_session()
        sweep_runner.end_run(snapshot)             # call after session ends
    """

    def __init__(self, sweep_dir: Path = None):
        self._lock = threading.Lock()
        self._active_run: Optional[dict] = None
        self._session_id: Optional[str] = None
        self._run_start: Optional[datetime] = None

        self._sweep_dir = sweep_dir if sweep_dir is not None else SWEEP_DIR
        self._sweep_dir.mkdir(exist_ok=True)
        self._state = SweepState(self._sweep_dir / "sweep_state.json")

        # JSONL output file — one per sweep campaign
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._jsonl_path = self._sweep_dir / f"sweep_{ts}.jsonl"

        total = TOTAL_RUNS
        remaining = max(0, total - self._state.run_index)
        logger.info(
            f"Sweep initialised: {remaining}/{total} runs remaining "
            f"({'ACTIVE' if SWEEP_ACTIVE else 'INACTIVE'})"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def next_run(self, session_id: str) -> dict:
        """
        Called at session start. Returns VAD config dict for this run.
        If sweep is inactive or complete, returns DEFAULT_VAD.
        """
        if not SWEEP_ACTIVE:
            return dict(DEFAULT_VAD)

        with self._lock:
            idx = self._state.run_index

            if idx >= TOTAL_RUNS:
                logger.info(
                    "Sweep complete — all design points exhausted. Returning defaults."
                )
                return dict(DEFAULT_VAD)

            design_point = CCD_MATRIX[idx]
            self._active_run = design_point
            self._session_id = session_id
            self._run_start = datetime.now()

            vad_config = {
                "threshold": design_point["threshold"],
                "silence_duration_ms": design_point["silence_duration_ms"],
                "prefix_padding_ms": design_point["prefix_padding_ms"],
            }

            logger.info(
                f"Sweep run {idx + 1}/{TOTAL_RUNS} "
                f"(point={design_point['design_point_index']}, "
                f"rep={design_point['rep']}): {vad_config}"
            )

            return vad_config

    def end_run(self, snapshot: dict):
        """
        Called at session end with metrics_collector.get_run_snapshot().
        Writes JSONL record and advances design matrix position.
        No-op if sweep is inactive.
        """
        if not SWEEP_ACTIVE:
            return

        with self._lock:
            if self._active_run is None:
                logger.warning("end_run() called but no active run — ignoring")
                return

            idx = self._state.run_index
            now = datetime.now()
            duration_s = (
                (now - self._run_start).total_seconds() if self._run_start else None
            )

            record = {
                "run_index": idx,
                "run_number": idx + 1,
                "total_runs": TOTAL_RUNS,
                "design_point_index": self._active_run["design_point_index"],
                "rep": self._active_run["rep"],
                "session_id": self._session_id,
                "started_at": self._run_start.isoformat() if self._run_start else None,
                "ended_at": now.isoformat(),
                "duration_s": round(duration_s, 1) if duration_s else None,
                "annotation": {
                    "valid": None,
                    "warmup": False,
                    "speaker": 1,
                },
                "snapshot": snapshot,
            }

            self._write_jsonl(record)
            self._state.advance()

            remaining = max(0, TOTAL_RUNS - (idx + 1))
            logger.info(
                f"Sweep run {idx + 1} saved "
                f"(FPR={snapshot.get('openai', {}).get('fpr', 'n/a')}, "
                f"latency={snapshot.get('openai', {}).get('response_latency_ms', 'n/a')}ms). "
                f"{remaining} runs remaining."
            )

            self._active_run = None
            self._session_id = None
            self._run_start = None

    @property
    def is_complete(self) -> bool:
        return SWEEP_ACTIVE and self._state.run_index >= TOTAL_RUNS

    @property
    def progress(self) -> tuple[int, int]:
        """Returns (completed_runs, total_runs)."""
        return self._state.run_index, TOTAL_RUNS

    def reset(self):
        """Reset sweep to run 1. Use with caution — does not delete existing JSONL."""
        self._state.reset()
        logger.warning("Sweep reset to run 1")

    def reset_metrics(self, metrics_collector) -> None:
        """
        Reset sweep accumulators in the metrics collector at run start.
        Call after next_run() and before connect_openai():

            vad = sweep_runner.next_run(session_id)
            sweep_runner.reset_metrics(metrics_collector)
            configure_session(case_config, vad)
            await connect_openai(...)
        """
        metrics_collector.reset_sweep_accumulators()
        logger.debug("Sweep accumulators reset for new run")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _write_jsonl(self, record: dict):
        try:
            with open(self._jsonl_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write sweep record: {e}")


# Singleton — import this in main.py
sweep_runner = SweepRunner()
