"""
Data verification for LLM-as-a-Judge ROC pipeline.

Checks that JSONL files contain all fields required for the two active ROC curves:

  ROC-1  VAD parameter sweep  — FPR vs. VAD config per session (sweep_*.jsonl)
  ROC-2  persona_adherence    — judge score per AI turn (transcript embedded in
                                sweep_*.jsonl, or standalone transcripts_*.jsonl)
  ROC-3  conversation_flow    — judge flow score per AI turn (same source as ROC-2)

Exit code 0 = all checked files are ready.
Exit code 1 = one or more fields missing or no files found.

Usage:
  python verify_data.py                          # scans example_data/ by default
  python verify_data.py path/to/sweep.jsonl ...  # explicit files
"""

import json
import sys
from pathlib import Path

# ── Field specs ───────────────────────────────────────────────────────────────
# Each entry: (display_name, getter)
# getter receives a raw JSONL record dict and returns the value (or None if missing)


def _get(record, *path):
    node = record
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


ROC1_VAD_FIELDS = [
    ("vad_config.threshold", lambda r: _get(r, "snapshot", "vad_config", "threshold")),
    (
        "vad_config.silence_duration_ms",
        lambda r: _get(r, "snapshot", "vad_config", "silence_duration_ms"),
    ),
    (
        "vad_config.prefix_padding_ms",
        lambda r: _get(r, "snapshot", "vad_config", "prefix_padding_ms"),
    ),
    ("openai.fpr", lambda r: _get(r, "snapshot", "openai", "fpr")),
    ("openai.vad_triggered", lambda r: _get(r, "snapshot", "openai", "vad_triggered")),
    (
        "openai.vad_false_triggers",
        lambda r: _get(r, "snapshot", "openai", "vad_false_triggers"),
    ),
    ("annotation.valid", lambda r: _get(r, "annotation", "valid")),
]


# For transcript records, the prompt/transcript may be top-level (transcripts_*.jsonl)
# or nested under snapshot (sweep_*.jsonl with embedded transcript)
def _transcript(r):
    return _get(r, "snapshot", "transcript") or _get(r, "transcript")


def _prompt(r):
    return _get(r, "snapshot", "prompt") or _get(r, "prompt")


ROC2_TRANSCRIPT_FIELDS = [
    ("prompt", _prompt),
    ("transcript", _transcript),
    ("session_id", lambda r: r.get("session_id")),
]

TRANSCRIPT_TURN_FIELDS = ["role", "text", "start", "end"]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_jsonl(path: Path) -> list[tuple[int, dict]]:
    records = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append((i, json.loads(line)))
            except json.JSONDecodeError as e:
                print(f"    [line {i}] JSON parse error: {e}")
    return records


def _check_fields(record, fields):
    return [name for name, getter in fields if getter(record) is None]


def _section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── Verification routines ─────────────────────────────────────────────────────


def verify_roc1(path: Path, records: list) -> bool:
    _section(f"ROC-1 VAD  ·  {path.name}")

    total = len(records)
    missing_counts = {name: 0 for name, _ in ROC1_VAD_FIELDS}
    first_bad: list[tuple[int, list]] = []

    for lineno, record in records:
        missing = _check_fields(record, ROC1_VAD_FIELDS)
        for m in missing:
            missing_counts[m] += 1
        if missing and len(first_bad) < 3:
            first_bad.append((lineno, missing))

    print(f"  Records : {total}")

    any_missing = any(c > 0 for c in missing_counts.values())
    if not any_missing:
        print("  Status  : OK — all VAD fields present in every record")
        return True

    print("  Status  : INCOMPLETE")
    for name, count in missing_counts.items():
        if count > 0:
            print(
                f"    MISSING  {name}  ({count}/{total} records, {100 * count // total}%)"
            )
    if first_bad:
        print(f"  First affected lines: {[ln for ln, _ in first_bad]}")
    return False


def verify_roc2(path: Path, records: list, source_label: str) -> bool:
    _section(f"ROC-2 persona_adherence  ·  {source_label}")

    total = len(records)
    no_prompt = 0
    no_transcript = 0
    incomplete_turns = 0
    total_ai_turns = 0
    sessions_with_data = 0

    for _, record in records:
        prompt = _prompt(record)
        transcript = _transcript(record)

        if not prompt:
            no_prompt += 1
        if not transcript:
            no_transcript += 1
            continue

        sessions_with_data += 1
        for turn in transcript:
            if turn.get("role") == "assistant":
                total_ai_turns += 1
                missing_turn_fields = [
                    f for f in TRANSCRIPT_TURN_FIELDS if not turn.get(f)
                ]
                if missing_turn_fields:
                    incomplete_turns += 1

    print(f"  Records              : {total}")
    print(f"  With transcript      : {sessions_with_data}/{total}")
    print(f"  With prompt          : {total - no_prompt}/{total}")
    print(f"  Total AI turns       : {total_ai_turns}  (judge inputs)")
    print(f"  Turns missing fields : {incomplete_turns}")

    ok = (
        no_transcript == 0
        and no_prompt == 0
        and incomplete_turns == 0
        and total_ai_turns > 0
    )

    if no_transcript == total:
        print("  Status  : INCOMPLETE — no transcript arrays found")
        print("            (requires transcript_logging=True in talkwise-pi config)")
    elif no_prompt > 0 or no_transcript > 0 or incomplete_turns > 0:
        print("  Status  : INCOMPLETE — see counts above")
    else:
        print("  Status  : OK — ready for judge.py")

    return ok


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    explicit = [Path(p) for p in sys.argv[1:]]
    if explicit:
        paths = explicit
    else:
        base = Path(__file__).parent.parent / "example_data"
        paths = sorted(base.glob("*.jsonl"))
        if not paths:
            print(f"No JSONL files in {base}. Pass file paths as arguments.")
            sys.exit(1)

    print("\nROC DATA VERIFICATION")
    print("  ROC-1  VAD parameter sweep    sweep_*.jsonl")
    print("  ROC-2  persona_adherence      transcript[s]_*.jsonl  or embedded in sweep")
    print(
        "  ROC-3  conversation_flow      judge flow score per AI turn (same source as ROC-2)"
    )

    sweep_paths = [p for p in paths if not p.stem.startswith("transcript")]
    transcript_paths = [p for p in paths if p.stem.startswith("transcript")]

    results: list[bool] = []

    for p in sweep_paths:
        records = _load_jsonl(p)
        if not records:
            print(f"\n  {p.name}: empty — skipping")
            continue

        results.append(verify_roc1(p, records))

        # New-format sweep files embed transcript in snapshot
        has_transcript = any(_transcript(r) is not None for _, r in records[:5])
        if has_transcript:
            results.append(verify_roc2(p, records, f"{p.name} (embedded)"))
        else:
            print(f"\n  NOTE: {p.name} has no embedded transcript")
            print("        Old-format data — only ROC-1 possible from this file")
            print(
                "        ROC-2 requires records produced with transcript_logging=True"
            )

    for p in transcript_paths:
        records = _load_jsonl(p)
        if not records:
            continue
        results.append(verify_roc2(p, records, p.name))

    print(f"\n{'═' * 60}")
    if not results:
        print("  No files checked.")
        sys.exit(1)
    all_ok = all(results)
    print(f"  OVERALL: {'ALL CHECKS PASSED' if all_ok else 'ISSUES FOUND — see above'}")
    print(f"{'═' * 60}\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
