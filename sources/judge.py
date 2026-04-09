"""
LLM-as-a-Judge for persona adherence and conversation flow in medical simulation transcripts.

Reads sweep or standalone transcript JSONL files and, for each assistant turn,
asks claude-sonnet-4-6 to evaluate two independent dimensions:

  1. Persona adherence   — did the AI stay in character as the patient?
  2. Conversation flow   — did the AI manage conversational transitions correctly?

Both judgments are captured in a single API call per turn.

Output JSONL adds a `judge` key to each record with the following structure:

  judge.turns[].persona_adherence_score  : float 0.0–1.0
  judge.turns[].label                    : TP | FP | TN | FN  (persona)
  judge.turns[].flow.flow_score          : float 0.0–1.0
  judge.turns[].flow.label               : TP | FP | TN | FN  (flow)
  judge.session_adherence_score          : float (mean persona score)
  judge.session_flow_score               : float (mean flow score)

Usage:
    python3 judge.py --input ../example_data/sweep_transcript_example.jsonl \\
                     --output judged.jsonl

Setup:
    export ANTHROPIC_API_KEY=<key>
    pip install anthropic
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Config ─────────────────────────────────────────────────────────────────────

CONFIG = {
    "model": os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6"),
    "max_tokens": 768,
    "temperature": 0.0,
    "max_retries": 3,
    "retry_base_delay_s": 2.0,
    "request_delay_s": 0.5,
}

# ── Judge prompt ────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a medical simulation evaluator. For each AI assistant turn you will
produce two independent judgments.

─── JUDGMENT 1: Persona Adherence ───────────────────────────────────────────

Evaluate whether the AI correctly maintained the patient persona defined in the
persona instruction.

  should_adhere (bool)
    True in almost all cases — the AI must always respond AS the patient.
    False only if the conversation context makes clear the simulation has
    formally ended and the AI is expected to speak as an assistant.

  did_adhere (bool)
    True if the AI stays in-character: lay vocabulary, correct emotional tone,
    first-person patient perspective.
    False if the AI exhibits knowledge the patient would not plausibly have
    (e.g. a lay patient answering with medical precision).

  persona_adherence_score (float 0.0–1.0)
    1.0 = perfect in-character response.
    0.0 = complete character break undermining the simulation.

  persona_reason (str)
    One sentence citing specific text evidence.

─── JUDGMENT 2: Conversation Flow ───────────────────────────────────────────

Evaluate whether the AI correctly managed the conversational transition at this
turn.

  should_transition (bool)
    True if a conversational redirect or topic change was needed at this turn.
    Examples: the user posed an off-topic or clinical question the patient
    should not engage with; the current topic was exhausted and a natural
    redirect was appropriate.
    False if the AI should simply continue the established topic thread.

  did_transition_appropriately (bool)
    True if the AI's conversational management matched what was needed:
    redirected when it should, or continued when it should.
    False if the AI answered an off-topic question instead of redirecting, or
    unnecessarily changed topic when continuation was appropriate.

  flow_score (float 0.0–1.0)
    1.0 = perfect conversational flow management.
    0.0 = complete failure (e.g. clinical question answered in full detail
          instead of redirected, or abrupt unexplained topic switch).

  flow_reason (str)
    One sentence citing specific text evidence.\
"""

_JUDGE_TOOL = {
    "name": "record_judgment",
    "description": "Record both the persona adherence and conversation flow judgment for this AI turn.",
    "input_schema": {
        "type": "object",
        "properties": {
            # ── Persona adherence ──
            "should_adhere": {
                "type": "boolean",
                "description": "True if the AI should respond in-character as the patient.",
            },
            "did_adhere": {
                "type": "boolean",
                "description": "True if the AI's response actually stayed in-character.",
            },
            "persona_adherence_score": {
                "type": "number",
                "description": "Continuous score 0.0–1.0 of persona maintenance quality.",
            },
            "persona_reason": {
                "type": "string",
                "description": "One sentence citing specific text evidence for the persona judgment.",
            },
            # ── Conversation flow ──
            "should_transition": {
                "type": "boolean",
                "description": "True if a conversational redirect or topic change was needed at this turn.",
            },
            "did_transition_appropriately": {
                "type": "boolean",
                "description": "True if the AI correctly managed the conversational transition.",
            },
            "flow_score": {
                "type": "number",
                "description": "Continuous score 0.0–1.0 of conversational flow management quality.",
            },
            "flow_reason": {
                "type": "string",
                "description": "One sentence citing specific text evidence for the flow judgment.",
            },
        },
        "required": [
            "should_adhere",
            "did_adhere",
            "persona_adherence_score",
            "persona_reason",
            "should_transition",
            "did_transition_appropriately",
            "flow_score",
            "flow_reason",
        ],
    },
}

# ── Data helpers ────────────────────────────────────────────────────────────────


def _get(record, *path):
    node = record
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _transcript(r):
    return _get(r, "snapshot", "transcript") or _get(r, "transcript")


def _prompt(r):
    return _get(r, "snapshot", "prompt") or _get(r, "prompt")


def _session_id(r):
    return r.get("session_id", "<unknown>")


# ── Judge logic ─────────────────────────────────────────────────────────────────


def _persona_label(should_adhere: bool, did_adhere: bool) -> str:
    if should_adhere and did_adhere:
        return "TP"
    if not should_adhere and did_adhere:
        return "FP"
    if not should_adhere and not did_adhere:
        return "TN"
    return "FN"


def _flow_label(should_transition: bool, did_transition_appropriately: bool) -> str:
    if should_transition and did_transition_appropriately:
        return "TP"
    if not should_transition and did_transition_appropriately:
        return "TN"
    if should_transition and not did_transition_appropriately:
        return "FN"
    return "FP"


def _format_user_message(persona_prompt: str, transcript: list, eval_index: int) -> str:
    lines = [f"Persona instruction:\n{persona_prompt}\n", "Conversation:"]
    for i, turn in enumerate(transcript[: eval_index + 1]):
        role = turn.get("role", "?")
        text = turn.get("text", "")
        tag = "[user]" if role == "user" else "[assistant]"
        marker = "  ← EVALUATE THIS TURN" if i == eval_index else ""
        lines.append(f"{tag}: {text}{marker}")
    return "\n".join(lines)


def _call_judge(client: anthropic.Anthropic, user_msg: str) -> dict:
    for attempt in range(CONFIG["max_retries"]):
        try:
            resp = client.messages.create(
                model=CONFIG["model"],
                max_tokens=CONFIG["max_tokens"],
                temperature=CONFIG["temperature"],
                system=_SYSTEM,
                tools=[_JUDGE_TOOL],
                tool_choice={"type": "tool", "name": "record_judgment"},
                messages=[{"role": "user", "content": user_msg}],
            )
            tool_block = next(b for b in resp.content if b.type == "tool_use")
            return tool_block.input
        except anthropic.RateLimitError:
            if attempt < CONFIG["max_retries"] - 1:
                delay = CONFIG["retry_base_delay_s"] * (2**attempt)
                print(
                    f"      rate-limited, retrying in {delay:.0f}s …", file=sys.stderr
                )
                time.sleep(delay)
            else:
                raise
        except Exception as exc:
            if attempt < CONFIG["max_retries"] - 1:
                delay = CONFIG["retry_base_delay_s"] * (2**attempt)
                print(
                    f"      error ({exc}), retrying in {delay:.0f}s …", file=sys.stderr
                )
                time.sleep(delay)
            else:
                raise


def judge_session(client: anthropic.Anthropic, record: dict) -> dict:
    persona_prompt = _prompt(record)
    transcript = _transcript(record)
    sid = _session_id(record)

    if not persona_prompt:
        print(f"  [{sid}] no persona prompt — skipping", file=sys.stderr)
        return {}
    if not transcript:
        print(f"  [{sid}] no transcript — skipping", file=sys.stderr)
        return {}

    turns_out = []
    assistant_indices = [
        i for i, t in enumerate(transcript) if t.get("role") == "assistant"
    ]

    for idx in assistant_indices:
        user_msg = _format_user_message(persona_prompt, transcript, idx)
        data = _call_judge(client, user_msg)
        time.sleep(CONFIG["request_delay_s"])

        # Persona
        should_adhere = bool(data.get("should_adhere", True))
        did_adhere = bool(data.get("did_adhere", True))
        persona_score = float(data.get("persona_adherence_score", 0.0))

        # Flow
        should_transition = bool(data.get("should_transition", False))
        did_transition = bool(data.get("did_transition_appropriately", True))
        flow_score = float(data.get("flow_score", 0.0))

        p_label = _persona_label(should_adhere, did_adhere)
        f_label = _flow_label(should_transition, did_transition)

        turns_out.append(
            {
                "turn_index": idx,
                # Persona adherence (flat, backward-compatible)
                "should_adhere": should_adhere,
                "did_adhere": did_adhere,
                "persona_adherence_score": persona_score,
                "label": p_label,
                "reason": data.get("persona_reason", ""),
                # Conversation flow (nested)
                "flow": {
                    "should_transition": should_transition,
                    "did_transition_appropriately": did_transition,
                    "flow_score": flow_score,
                    "label": f_label,
                    "reason": data.get("flow_reason", ""),
                },
            }
        )
        print(
            f"  [{sid}] turn {idx:2d}  persona={p_label} score={persona_score:.2f}"
            f"  flow={f_label} score={flow_score:.2f}",
            file=sys.stderr,
        )

    if not turns_out:
        return {}

    session_adherence = sum(t["persona_adherence_score"] for t in turns_out) / len(
        turns_out
    )
    session_flow = sum(t["flow"]["flow_score"] for t in turns_out) / len(turns_out)

    return {
        "model": CONFIG["model"],
        "judged_at": datetime.now(timezone.utc).isoformat(),
        "turns": turns_out,
        "session_adherence_score": round(session_adherence, 4),
        "session_flow_score": round(session_flow, 4),
    }


# ── IO ──────────────────────────────────────────────────────────────────────────


def _load_jsonl(path: Path) -> list:
    records = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  {path.name}:{lineno}: JSON error — {exc}", file=sys.stderr)
    return records


def _resolve_inputs(raw: list) -> list:
    paths = []
    for r in raw:
        p = Path(r)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.jsonl")))
        else:
            paths.append(p)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge: score persona adherence and conversation flow per AI turn."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        metavar="PATH",
        help="JSONL file(s) or directory containing JSONL files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="FILE",
        help="Output JSONL file (annotated records).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    input_paths = _resolve_inputs(args.input)

    if not input_paths:
        print("error: no JSONL files found", file=sys.stderr)
        sys.exit(1)

    all_records = []
    for p in input_paths:
        print(f"loading {p}", file=sys.stderr)
        all_records.extend(_load_jsonl(p))

    print(f"judging {len(all_records)} session(s) …", file=sys.stderr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as fh:
        for i, record in enumerate(all_records, 1):
            sid = _session_id(record)
            print(f"[{i}/{len(all_records)}] {sid}", file=sys.stderr)
            judge_result = judge_session(client, record)
            annotated = dict(record)
            if judge_result:
                annotated["judge"] = judge_result
            fh.write(json.dumps(annotated, ensure_ascii=False) + "\n")

    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
