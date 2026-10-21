"""
Regenerate the Paper 1 figures and verify the manuscript's reported values.

Runs the canonical pipeline (sources/analyse_sweep.py) on the study data in
results/ and writes Fig1-Fig4 (.eps, .tif, .png), summary.txt and
pareto_model.json to papers/paper1/figures/. Exits non-zero if any value
reported in the manuscript cannot be reproduced at its reported precision.

Usage (from the repository root):
    python3 papers/paper1/make_figures.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sources"))

import analyse_sweep

# Manuscript values: (label, function of results, reported value, tolerance).
# Tolerances are half a unit of the last reported digit. RSM coefficients are
# from the models including the recording-day covariate unless noted.
REFERENCE = [
    ("N sessions", lambda r: len(r["df"]), 76, 0),
    ("design points", lambda r: r["df"]["design_point"].nunique(), 19, 0),
    ("SD threshold", lambda r: r["scaling"]["threshold"][1], 0.164, 0.0005),
    ("SD silence (ms)", lambda r: r["scaling"]["silence_ms"][1], 310.4, 0.05),
    ("SD prefix (ms)", lambda r: r["scaling"]["prefix_ms"][1], 109.5, 0.05),
    ("FPR min", lambda r: r["df"]["fpr"].min(), 0.000, 0.0005),
    ("FPR max", lambda r: r["df"]["fpr"].max(), 0.444, 0.0005),
    ("FPR mean", lambda r: r["df"]["fpr"].mean(), 0.165, 0.0005),
    ("FPR SD", lambda r: r["df"]["fpr"].std(), 0.096, 0.0005),
    ("TL min (ms)", lambda r: r["df"]["tl_ms"].min(), 960, 0.5),
    ("TL max (ms)", lambda r: r["df"]["tl_ms"].max(), 2602, 0.5),
    ("TL mean (ms)", lambda r: r["df"]["tl_ms"].mean(), 1552, 0.5),
    ("TL SD (ms)", lambda r: r["df"]["tl_ms"].std(), 353, 0.5),
    ("WebSocket RTT mean", lambda r: r["df"]["ws_latency_ms"].mean(), 207.0, 0.05),
    ("WebSocket RTT max", lambda r: r["df"]["ws_latency_ms"].max(), 276.1, 0.05),
    ("TDP mean", lambda r: 1 - r["df"]["fpr"].mean(), 0.835, 0.0005),
    ("FPR R²", lambda r: r["models"]["fpr"].rsquared, 0.353, 0.0005),
    ("FPR b1", lambda r: r["models"]["fpr"].params["x1"], -0.026, 0.0005),
    ("FPR p1", lambda r: r["models"]["fpr"].pvalues["x1"], 0.009, 0.0005),
    ("FPR b2", lambda r: r["models"]["fpr"].params["x2"], -0.038, 0.0005),
    ("FPR b3", lambda r: r["models"]["fpr"].params["x3"], -0.006, 0.0005),
    ("FPR p3", lambda r: r["models"]["fpr"].pvalues["x3"], 0.533, 0.0005),
    ("FPR b13", lambda r: r["models"]["fpr"].params["x1:x3"], 0.015, 0.0005),
    ("FPR p13", lambda r: r["models"]["fpr"].pvalues["x1:x3"], 0.054, 0.0005),
    ("FPR b_day", lambda r: r["models"]["fpr"].params["day"], 0.002, 0.0005),
    ("FPR p_day", lambda r: r["models"]["fpr"].pvalues["day"], 0.91, 0.005),
    ("TL R²", lambda r: r["models"]["tl_ms"].rsquared, 0.765, 0.0005),
    ("TL b2 (ms/SD)", lambda r: r["models"]["tl_ms"].params["x2"], 302.4, 0.05),
    ("TL b_day (ms)", lambda r: r["models"]["tl_ms"].params["day"], -94.9, 0.05),
    ("TL p_day", lambda r: r["models"]["tl_ms"].pvalues["day"], 0.03, 0.005),
    (
        "TL R² without day",
        lambda r: r["models_noday"]["tl_ms"].rsquared,
        0.747,
        0.0005,
    ),
    (
        "TL b2 without day (ms/SD)",
        lambda r: r["models_noday"]["tl_ms"].params["x2"],
        300.7,
        0.05,
    ),
    (
        "speech_stopped R² without day",
        lambda r: r["models_noday"]["response_latency_ms"].rsquared,
        0.079,
        0.0005,
    ),
    (
        "speech_stopped b2 without day",
        lambda r: r["models_noday"]["response_latency_ms"].params["x2"],
        -9.7,
        0.05,
    ),
    (
        "speech_stopped p2 without day",
        lambda r: r["models_noday"]["response_latency_ms"].pvalues["x2"],
        0.66,
        0.005,
    ),
    (
        "r(silence, IR)",
        lambda r: np.corrcoef(
            r["df"]["silence_ms"],
            r["df"]["interruptions"] / (r["df"]["duration_s"] / 60),
        )[0, 1],
        -0.44,
        0.005,
    ),
    (
        "r(silence, TDP)",
        lambda r: np.corrcoef(r["df"]["silence_ms"], 1 - r["df"]["fpr"])[0, 1],
        0.40,
        0.005,
    ),
]
P_BELOW = [
    ("FPR p2", lambda r: r["models"]["fpr"].pvalues["x2"], 0.001),
    ("TL p2", lambda r: r["models"]["tl_ms"].pvalues["x2"], 0.001),
]
# (threshold, silence, prefix) -> (mean FPR, SD)
DESIGN_POINT_FPR = {
    (0.625, 775, 300): (0.126, 0.081),
    (0.85, 1200, 150): (0.067, 0.099),
}
SESSION_FRONT = [(0.625, 350, 300, 0.056, 960), (0.625, 775, 300, 0.000, 1446)]


def check(results: dict) -> list[str]:
    failed = []
    print("\nManuscript reference check:")
    for label, fn, expected, tol in REFERENCE:
        value = fn(results)
        ok = abs(value - expected) <= tol
        print(
            f"  {'OK ' if ok else 'DEV'} {label:28s} {value:12.4f}  reported {expected}"
        )
        if not ok:
            failed.append(label)
    for label, fn, bound in P_BELOW:
        value = fn(results)
        ok = value < bound
        print(
            f"  {'OK ' if ok else 'DEV'} {label:28s} {value:12.2e}  reported < {bound}"
        )
        if not ok:
            failed.append(label)

    df = results["df"]
    for (t, s, p), (mean, sd) in DESIGN_POINT_FPR.items():
        v = df.loc[
            np.isclose(df["threshold"], t)
            & (df["silence_ms"] == s)
            & (df["prefix_ms"] == p),
            "fpr",
        ]
        ok = round(v.mean(), 3) == mean and round(v.std(), 3) == sd
        print(
            f"  {'OK ' if ok else 'DEV'} FPR at ({t:g}, {s}, {p}){'':7s}"
            f"{v.mean():.3f} ({v.std():.3f})  reported {mean} ({sd})"
        )
        if not ok:
            failed.append(f"design point ({t:g}, {s}, {p})")

    front = sorted(
        (r.threshold, r.silence_ms, r.prefix_ms, round(r.fpr, 3), round(r.tl_ms))
        for r in results["session_front"].itertuples()
    )
    ok = front == SESSION_FRONT
    print(f"  {'OK ' if ok else 'DEV'} session-level front          {front}")
    if not ok:
        failed.append("session-level front")
    return failed


def main():
    results = analyse_sweep.run(ROOT / "results", Path(__file__).parent / "figures")
    failed = check(results)
    if failed:
        sys.exit(f"\nValues deviate from the manuscript: {failed}")
    print("\nAll manuscript values reproduced.")


if __name__ == "__main__":
    main()
