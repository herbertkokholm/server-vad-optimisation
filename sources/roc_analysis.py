"""
ROC curve generation for LLM-as-a-Judge evaluation of medical simulation transcripts.

Produces four publication-quality figures:

  fig1_roc_vad.png      — ROC-1: VAD trigger classification.
  fig2_roc_persona.png  — ROC-2: persona adherence score as discriminator.
  fig3_roc_flow.png     — ROC-3: conversation flow score as discriminator.
  fig4_roc_combined.png — All active ROC curves on one axes, annotated with
                          Cronbach's alpha (internal consistency of judge scores).

Each figure shows:
  - ROC curve with 95 % bootstrap CI in the legend label
  - Optimal operating point (Youden's J = argmax TPR − FPR) marked as a dot
  - Chance diagonal

ROC-1 note: the curve is constructed by sweeping the VAD threshold parameter
and recording aggregate session-level trigger counts at each operating point.
It is an operating-point ROC, not an individual-prediction ROC; this should
be stated explicitly when reporting results.

ROC-1: y_true = 1 for true speech triggers, 0 for false alarms; y_score = threshold.
ROC-2: y_true = 1 if should_adhere; y_score = persona_adherence_score.
ROC-3: y_true = 1 if did_transition_appropriately; y_score = flow_score.

Cronbach's alpha is computed from the matrix of per-turn judge scores
(persona + flow, and uncertainty when available). VAD is excluded because it
operates at session/threshold level rather than per turn.

Usage:
    python3 roc_analysis.py --sweep ../example_data/sweep_transcript_example.jsonl \\
                             --judged judged.jsonl --output figures/

    # ROC-1 only:
    python3 roc_analysis.py --sweep sweep.jsonl --output figures/

    # ROC-2, ROC-3 and combined only:
    python3 roc_analysis.py --judged judged.jsonl --output figures/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

# ── Config ─────────────────────────────────────────────────────────────────────

CONFIG = {
    "dpi": 300,
    "figsize": (7, 5),
    "color_roc": "#1a1a6e",
    "color_diag": "#999999",
    "color_grid": "#cccccc",
    "colors": {
        "roc1": "#1a1a6e",  # navy    — VAD
        "roc2": "#8b1a1a",  # crimson — persona adherence
        "roc3": "#1a6e2e",  # forest  — conversation flow
        "roc4": "#6e4a1a",  # sienna  — uncertainty (future)
    },
    "n_bootstrap": 1000,
    "ci_level": 0.95,
    "bootstrap_seed": 42,
}

# ── Style ───────────────────────────────────────────────────────────────────────


def set_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": CONFIG["dpi"],
            "axes.grid": True,
            "grid.color": CONFIG["color_grid"],
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# ── Data helpers ────────────────────────────────────────────────────────────────


def _get(record, *path):
    node = record
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


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


def _load_from_paths(paths) -> list:
    if not paths:
        return []
    records = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            for f in sorted(pp.glob("*.jsonl")):
                records.extend(_load_jsonl(f))
        else:
            records.extend(_load_jsonl(pp))
    return records


# ── ROC data builders ───────────────────────────────────────────────────────────


def build_vad_roc_data(records: list) -> tuple:
    """Build operating-point ROC dataset for ROC-1 (VAD).

    Each session at threshold τ contributes:
      (vad_triggered - vad_false_triggers) true-trigger events  with score = τ
      vad_false_triggers false-trigger events                   with score = τ

    This is an operating-point ROC: all events within a session share the same
    score (the VAD threshold). The curve shows how system-level FPR/TPR trade off
    as τ is varied across sessions, not individual trigger-level discrimination.
    This distinction must be stated explicitly when reporting results.

    A higher τ is more conservative; higher score → more selective → trigger
    events more likely genuine speech.
    """
    y_true, y_score = [], []
    skipped = 0
    for r in records:
        threshold = _get(r, "snapshot", "vad_config", "threshold")
        vad_triggered = _get(r, "snapshot", "openai", "vad_triggered") or 0
        vad_false_triggers = _get(r, "snapshot", "openai", "vad_false_triggers") or 0

        if threshold is None or vad_triggered == 0:
            skipped += 1
            continue

        n_true = max(0, int(vad_triggered) - int(vad_false_triggers))
        n_false = int(vad_false_triggers)

        y_true.extend([1] * n_true + [0] * n_false)
        y_score.extend([float(threshold)] * (n_true + n_false))

    if skipped:
        print(
            f"  ROC-1: skipped {skipped} session(s) (missing threshold or zero triggers)",
            file=sys.stderr,
        )
    return np.array(y_true, dtype=float), np.array(y_score, dtype=float)


def build_adherence_roc_data(records: list) -> tuple:
    """Build per-turn binary dataset for ROC-2 (persona adherence).

    y_true = 1 if should_adhere else 0
    y_score = persona_adherence_score
    """
    y_true, y_score = [], []
    sessions_with_judge = 0
    for r in records:
        judge = r.get("judge")
        if not judge:
            continue
        sessions_with_judge += 1
        for turn in judge.get("turns", []):
            should_adhere = turn.get("should_adhere")
            score = turn.get("persona_adherence_score")
            if should_adhere is None or score is None:
                continue
            y_true.append(1 if should_adhere else 0)
            y_score.append(float(score))

    print(
        f"  ROC-2: {sessions_with_judge} session(s), {len(y_true)} turn(s)",
        file=sys.stderr,
    )
    return np.array(y_true, dtype=float), np.array(y_score, dtype=float)


def build_flow_roc_data(records: list) -> tuple:
    """Build per-turn binary dataset for ROC-3 (conversation flow).

    y_true = 1 if did_transition_appropriately else 0
    y_score = flow_score

    flow_score measures quality of execution, not whether a transition was
    contextually required. It therefore correlates with did_transition_appropriately
    (correct behaviour = high score) rather than with should_transition. Using
    should_transition as y_true would invert the curve (AUC < 0.5).

    Only sessions produced by the extended judge.py (with flow annotations) contribute.
    Sessions with no flow data are silently skipped.
    """
    y_true, y_score = [], []
    sessions_with_flow = 0
    for r in records:
        judge = r.get("judge")
        if not judge:
            continue
        has_flow = False
        for turn in judge.get("turns", []):
            flow = turn.get("flow")
            if not flow:
                continue
            did_transition = flow.get("did_transition_appropriately")
            score = flow.get("flow_score")
            if did_transition is None or score is None:
                continue
            has_flow = True
            y_true.append(1 if did_transition else 0)
            y_score.append(float(score))
        if has_flow:
            sessions_with_flow += 1

    print(
        f"  ROC-3: {sessions_with_flow} session(s), {len(y_true)} turn(s)",
        file=sys.stderr,
    )
    return np.array(y_true, dtype=float), np.array(y_score, dtype=float)


# ── Statistical helpers ─────────────────────────────────────────────────────────


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray) -> tuple:
    """Compute 95 % bootstrap confidence interval for AUC (non-parametric).

    Returns (lower, upper) or (nan, nan) if too few valid bootstrap samples.
    """
    rng = np.random.RandomState(CONFIG["bootstrap_seed"])
    n = len(y_true)
    aucs = []
    for _ in range(CONFIG["n_bootstrap"]):
        idx = rng.randint(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        f, t, _ = roc_curve(yt, ys)
        aucs.append(auc(f, t))
    if len(aucs) < 10:
        return (float("nan"), float("nan"))
    lo = (1.0 - CONFIG["ci_level"]) / 2.0 * 100
    hi = (1.0 + CONFIG["ci_level"]) / 2.0 * 100
    return (float(np.percentile(aucs, lo)), float(np.percentile(aucs, hi)))


def youden_optimal(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> dict:
    """Return the optimal operating point via Youden's J = argmax(TPR − FPR)."""
    j = tpr - fpr
    idx = int(np.argmax(j))
    return {
        "fpr": float(fpr[idx]),
        "tpr": float(tpr[idx]),
        "threshold": float(thresholds[idx]),
        "j": float(j[idx]),
    }


# ── Cronbach's alpha ────────────────────────────────────────────────────────────


def compute_cronbach_alpha(scores: np.ndarray) -> float:
    """Compute Cronbach's alpha for internal consistency across judge score dimensions.

    Args:
        scores: (n_turns, n_items) array. Each column is one judge dimension
                (e.g. persona_adherence_score, flow_score).

    Returns:
        alpha as float, or nan if not computable (< 2 items or zero total variance).
    """
    n, k = scores.shape
    if k < 2 or n < 2:
        return float("nan")
    item_variances = scores.var(axis=0, ddof=1)
    total_variance = scores.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return float("nan")
    return float((k / (k - 1)) * (1.0 - item_variances.sum() / total_variance))


def build_judge_scores_matrix(records: list) -> tuple:
    """Extract per-turn judge scores for Cronbach's alpha computation.

    Returns (scores, labels) where:
      scores : (n_turns, n_items) float array
      labels : list of column names

    VAD is excluded — it operates at session/threshold level, not per turn.
    Turns missing any score are dropped to keep the matrix rectangular.
    """
    rows = []
    for r in records:
        judge = r.get("judge")
        if not judge:
            continue
        for turn in judge.get("turns", []):
            persona = turn.get("persona_adherence_score")
            flow = turn.get("flow", {}).get("flow_score")
            if persona is None or flow is None:
                continue
            row = [float(persona), float(flow)]
            uncertainty = turn.get("uncertainty_score")
            if uncertainty is not None:
                row.append(float(uncertainty))
            rows.append(row)

    if not rows:
        return np.empty((0, 0)), []

    matrix = np.array(rows, dtype=float)
    labels = ["persona", "flow"] + (["uncertainty"] if matrix.shape[1] == 3 else [])
    return matrix, labels


# ── Plot helpers ────────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, path: Path):
    fig.savefig(path, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}", file=sys.stderr)


def _draw_roc_curve(ax, fpr, tpr, roc_auc, ci, opt, label, color):
    """Draw a single ROC curve with CI legend label and Youden's J marker."""
    ci_str = f" [{ci[0]:.3f}–{ci[1]:.3f}]" if ci and not np.isnan(ci[0]) else ""
    ax.plot(
        fpr,
        tpr,
        color=color,
        lw=1.5,
        label=f"{label} (AUC = {roc_auc:.3f}{ci_str})",
    )
    if opt:
        ax.scatter(
            [opt["fpr"]],
            [opt["tpr"]],
            color=color,
            s=50,
            zorder=5,
            edgecolors="white",
            linewidths=0.8,
        )
        ax.annotate(
            f"τ* = {opt['threshold']:.3f}",
            xy=(opt["fpr"], opt["tpr"]),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )


def _finish_ax(ax):
    ax.plot(
        [0, 1],
        [0, 1],
        color=CONFIG["color_diag"],
        lw=0.8,
        linestyle="--",
        label="Chance",
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)


# ── Individual ROC figures ──────────────────────────────────────────────────────


def plot_roc1(fpr, tpr, roc_auc, ci, opt, out_dir: Path):
    fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    _draw_roc_curve(ax, fpr, tpr, roc_auc, ci, opt, "VAD ROC", CONFIG["colors"]["roc1"])
    _finish_ax(ax)
    ax.set_title("ROC-1 — VAD Trigger Classification")
    ax.text(
        0.02,
        0.02,
        "Operating-point ROC: score = VAD threshold τ (not per-event prediction).",
        transform=ax.transAxes,
        fontsize=6.5,
        color="#666666",
        va="bottom",
    )
    _save(fig, out_dir / "fig1_roc_vad.png")


def plot_roc2(fpr, tpr, roc_auc, ci, opt, out_dir: Path):
    fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    _draw_roc_curve(
        ax,
        fpr,
        tpr,
        roc_auc,
        ci,
        opt,
        "Persona adherence ROC",
        CONFIG["colors"]["roc2"],
    )
    _finish_ax(ax)
    ax.set_title("ROC-2 — Dialogue Policy (Persona Adherence)")
    _save(fig, out_dir / "fig2_roc_persona.png")


def plot_roc3(fpr, tpr, roc_auc, ci, opt, out_dir: Path):
    fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    _draw_roc_curve(
        ax,
        fpr,
        tpr,
        roc_auc,
        ci,
        opt,
        "Conversation flow ROC",
        CONFIG["colors"]["roc3"],
    )
    _finish_ax(ax)
    ax.set_title("ROC-3 — Conversation Flow Management")
    _save(fig, out_dir / "fig3_roc_flow.png")


# ── Combined ROC figure ─────────────────────────────────────────────────────────


def plot_roc_combined(curves: list, alpha: float, n_turns: int, out_dir: Path):
    """Plot all active ROC curves with CI labels, Youden's J markers, and Cronbach's alpha.

    Each entry in curves must have:
      label, color, fpr, tpr, roc_auc, ci (tuple), opt (dict or None)
    """
    fig, ax = plt.subplots(figsize=CONFIG["figsize"])

    for c in curves:
        _draw_roc_curve(
            ax,
            c["fpr"],
            c["tpr"],
            c["roc_auc"],
            c["ci"],
            c["opt"],
            c["label"],
            c["color"],
        )
    _finish_ax(ax)
    ax.legend(loc="lower right", bbox_to_anchor=(0.98, 0.25))

    if not np.isnan(alpha):
        judge_dims = ", ".join(
            c["label"].split(":")[0].strip() for c in curves if "VAD" not in c["label"]
        )
        alpha_text = f"Cronbach's α = {alpha:.3f}  (n = {n_turns} turns)"
        note = f"(judge dimensions: {judge_dims})"
    else:
        alpha_text = "Cronbach's α — insufficient judge dimensions"
        note = "(requires ≥ 2 judge score dimensions)"

    ax.text(
        0.98,
        0.18,
        f"{alpha_text}\n{note}",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.9,
        },
    )

    ax.set_title("Combined ROC — All Decision Boundaries")
    _save(fig, out_dir / "fig4_roc_combined.png")


# ── Main ────────────────────────────────────────────────────────────────────────


def _compute_roc(y_true, y_score, label):
    """Compute ROC curve, AUC, bootstrap CI and Youden's J. Print summary."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ci = bootstrap_auc_ci(y_true, y_score)
    opt = youden_optimal(fpr, tpr, thresholds)
    ci_str = f"[{ci[0]:.3f}–{ci[1]:.3f}]" if not np.isnan(ci[0]) else "[CI n/a]"
    print(
        f"  {label}: AUC = {roc_auc:.3f} {ci_str}  "
        f"Youden J = {opt['j']:.3f} at τ* = {opt['threshold']:.3f}",
        file=sys.stderr,
    )
    return fpr, tpr, roc_auc, ci, opt


def main():
    parser = argparse.ArgumentParser(
        description="Generate ROC-1 (VAD), ROC-2 (persona), ROC-3 (flow), and combined figures."
    )
    parser.add_argument(
        "--sweep",
        nargs="+",
        metavar="PATH",
        help="Sweep JSONL file(s) or directory for ROC-1.",
    )
    parser.add_argument(
        "--judged",
        nargs="+",
        metavar="PATH",
        help="Judged JSONL file(s) or directory for ROC-2 and ROC-3.",
    )
    parser.add_argument(
        "--output",
        default=".",
        metavar="DIR",
        help="Output directory for figure PNGs (default: current directory).",
    )
    args = parser.parse_args()

    if not args.sweep and not args.judged:
        parser.error("supply at least one of --sweep or --judged")

    set_style()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_curves = []

    if args.sweep:
        sweep_records = _load_from_paths(args.sweep)
        print(f"ROC-1: {len(sweep_records)} sweep session(s)", file=sys.stderr)
        y_true, y_score = build_vad_roc_data(sweep_records)
        if len(y_true) < 2 or len(np.unique(y_true)) < 2:
            print("  ROC-1: insufficient data — skipping", file=sys.stderr)
        else:
            fpr, tpr, roc_auc, ci, opt = _compute_roc(y_true, y_score, "ROC-1")
            plot_roc1(fpr, tpr, roc_auc, ci, opt, out_dir)
            combined_curves.append(
                {
                    "label": "ROC-1: VAD",
                    "color": CONFIG["colors"]["roc1"],
                    "fpr": fpr,
                    "tpr": tpr,
                    "roc_auc": roc_auc,
                    "ci": ci,
                    "opt": opt,
                }
            )

    if args.judged:
        judged_records = _load_from_paths(args.judged)

        print(f"ROC-2: {len(judged_records)} record(s)", file=sys.stderr)
        y_true, y_score = build_adherence_roc_data(judged_records)
        if len(y_true) < 2 or len(np.unique(y_true)) < 2:
            print("  ROC-2: insufficient data — skipping", file=sys.stderr)
        else:
            fpr, tpr, roc_auc, ci, opt = _compute_roc(y_true, y_score, "ROC-2")
            plot_roc2(fpr, tpr, roc_auc, ci, opt, out_dir)
            combined_curves.append(
                {
                    "label": "ROC-2: Persona",
                    "color": CONFIG["colors"]["roc2"],
                    "fpr": fpr,
                    "tpr": tpr,
                    "roc_auc": roc_auc,
                    "ci": ci,
                    "opt": opt,
                }
            )

        print(f"ROC-3: {len(judged_records)} record(s)", file=sys.stderr)
        y_true, y_score = build_flow_roc_data(judged_records)
        if len(y_true) < 2 or len(np.unique(y_true)) < 2:
            print("  ROC-3: insufficient data — skipping", file=sys.stderr)
        else:
            fpr, tpr, roc_auc, ci, opt = _compute_roc(y_true, y_score, "ROC-3")
            plot_roc3(fpr, tpr, roc_auc, ci, opt, out_dir)
            combined_curves.append(
                {
                    "label": "ROC-3: Flow",
                    "color": CONFIG["colors"]["roc3"],
                    "fpr": fpr,
                    "tpr": tpr,
                    "roc_auc": roc_auc,
                    "ci": ci,
                    "opt": opt,
                }
            )

        scores_matrix, score_labels = build_judge_scores_matrix(judged_records)
        alpha = (
            compute_cronbach_alpha(scores_matrix)
            if scores_matrix.size
            else float("nan")
        )
        n_turns = scores_matrix.shape[0] if scores_matrix.size else 0
        if not np.isnan(alpha):
            print(
                f"  Cronbach's α = {alpha:.3f}  ({n_turns} turns, items: {score_labels})",
                file=sys.stderr,
            )
        else:
            print("  Cronbach's α — insufficient data", file=sys.stderr)

        if len(combined_curves) >= 2:
            plot_roc_combined(combined_curves, alpha, n_turns, out_dir)
        elif combined_curves:
            print("  combined figure skipped (only one active ROC)", file=sys.stderr)


if __name__ == "__main__":
    main()
