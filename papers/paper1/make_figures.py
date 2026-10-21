"""
Regenerate the Paper 1 figures and verify the manuscript's reported values.

Runs the canonical pipeline (sources/analyse_sweep.py) on the study data in
results/ and writes Fig1-Fig4 (.eps, .tif, .png), summary.txt and
pareto_model.json to papers/paper1/figures/, then the Appendix A robustness
analyses (robustness.py) to robustness.txt. Exits non-zero if any value
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
import robustness

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
    ("WebSocket RTT SD", lambda r: r["df"]["ws_latency_ms"].std(), 14.3, 0.05),
    ("WebSocket RTT max", lambda r: r["df"]["ws_latency_ms"].max(), 276.1, 0.05),
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
        "speech_stopped R²",
        lambda r: r["models"]["response_latency_ms"].rsquared,
        0.145,
        0.0005,
    ),
    (
        "speech_stopped b2",
        lambda r: r["models"]["response_latency_ms"].params["x2"],
        -8.0,
        0.05,
    ),
    (
        "speech_stopped p2",
        lambda r: r["models"]["response_latency_ms"].pvalues["x2"],
        0.71,
        0.005,
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
# Non-dominated configuration means (sessions pooled per configuration) ->
# bootstrap non-dominance frequency
CONFIG_FRONT = {
    (0.85, 1200, 150): 0.66,
    (0.40, 1200, 450): 0.33,
    (0.85, 775, 300): 0.94,
    (0.85, 350, 450): 0.80,
}


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

    configs = results["configs"]
    front = configs[configs["non_dominated"]]
    found = {(r.threshold, r.silence_ms, r.prefix_ms) for r in front.itertuples()}
    ok = found == set(CONFIG_FRONT)
    print(f"  {'OK ' if ok else 'DEV'} configuration front          {sorted(found)}")
    if not ok:
        failed.append("configuration front")
    for r in front.itertuples():
        key = (r.threshold, r.silence_ms, r.prefix_ms)
        expected = CONFIG_FRONT.get(key)
        ok = expected is not None and abs(r.p_non_dominated - expected) <= 0.005
        print(
            f"  {'OK ' if ok else 'DEV'} P(non-dominated) {key}  "
            f"{r.p_non_dominated:.3f}  reported {expected}"
        )
        if not ok:
            failed.append(f"P(non-dominated) {key}")
    return failed


# Appendix A robustness values: (label, function of robustness values,
# reported value, tolerance)
ROBUSTNESS = [
    ("ICC FPR", lambda v: v["fpr.icc"], 0.014, 0.0005),
    ("ICC TL", lambda v: v["tl_ms.icc"], 0.000, 0.0005),
    ("lack of fit FPR F", lambda v: v["fpr.lof"][0], 0.67, 0.005),
    ("lack of fit FPR p", lambda v: v["fpr.lof"][1], 0.65, 0.005),
    ("lack of fit TL F", lambda v: v["tl_ms.lof"][0], 1.21, 0.005),
    ("lack of fit TL p", lambda v: v["tl_ms.lof"][1], 0.32, 0.005),
    ("GLM Pearson dispersion", lambda v: v["glm_dispersion"], 1.02, 0.005),
    ("GLM OR threshold", lambda v: v["glm"].loc["Threshold", "OR per SD"], 0.84, 0.005),
    (
        "GLM p threshold",
        lambda v: v["glm"].loc["Threshold", "p binomial"],
        0.012,
        0.0005,
    ),
    ("GLM OR silence", lambda v: v["glm"].loc["Silence", "OR per SD"], 0.78, 0.005),
    ("GLM OR prefix", lambda v: v["glm"].loc["Prefix", "OR per SD"], 0.98, 0.005),
    ("GLM p prefix", lambda v: v["glm"].loc["Prefix", "p binomial"], 0.807, 0.0005),
    (
        "GLM OR thr x prefix",
        lambda v: v["glm"].loc["Thr×Pre", "OR per SD"],
        1.10,
        0.005,
    ),
    (
        "GLM p thr x prefix",
        lambda v: v["glm"].loc["Thr×Pre", "p binomial"],
        0.087,
        0.0005,
    ),
    ("mixed FPR p threshold", lambda v: v["fpr.mixed_p"]["x1"], 0.009, 0.0005),
    ("mixed FPR p prefix", lambda v: v["fpr.mixed_p"]["x3"], 0.540, 0.0005),
    ("mixed FPR p thr x prefix", lambda v: v["fpr.mixed_p"]["x1:x3"], 0.055, 0.0005),
    ("mixed TL p threshold", lambda v: v["tl_ms.mixed_p"]["x1"], 0.125, 0.0005),
    ("19 means FPR p threshold", lambda v: v["fpr.means19_p"]["x1"], 0.038, 0.0005),
    ("19 means FPR p silence", lambda v: v["fpr.means19_p"]["x2"], 0.006, 0.0005),
    ("19 means FPR p prefix", lambda v: v["fpr.means19_p"]["x3"], 0.585, 0.0005),
    (
        "19 means FPR p thr x prefix",
        lambda v: v["fpr.means19_p"]["x1:x3"],
        0.109,
        0.0005,
    ),
    ("19 means TL p threshold", lambda v: v["tl_ms.means19_p"]["x1"], 0.098, 0.0005),
    ("triggers per session min", lambda v: v["triggers"].min(), 15, 0),
    ("triggers per session max", lambda v: v["triggers"].max(), 27, 0),
    ("triggers per session median", lambda v: v["triggers"].median(), 19, 0),
]
ROBUSTNESS_P_BELOW = [
    ("GLM p silence", lambda v: v["glm"].loc["Silence", "p binomial"], 0.001),
    ("mixed FPR p silence", lambda v: v["fpr.mixed_p"]["x2"], 0.001),
    ("mixed TL p silence", lambda v: v["tl_ms.mixed_p"]["x2"], 0.001),
    ("19 means TL p silence", lambda v: v["tl_ms.means19_p"]["x2"], 0.001),
]
# Configuration 0.85/350/150 is dominated, with this bootstrap frequency
DOMINATED_CONFIG = ((0.85, 350, 150), 0.52)
# Sessions per replication and recording date
REP_BY_DAY = {
    1: {"2026-04-23": 19, "2026-04-27": 0},
    2: {"2026-04-23": 19, "2026-04-27": 0},
    3: {"2026-04-23": 1, "2026-04-27": 18},
    4: {"2026-04-23": 0, "2026-04-27": 19},
}


def check_robustness(results: dict, values: dict) -> list[str]:
    failed = []
    print("\nAppendix A robustness check:")
    for label, fn, expected, tol in ROBUSTNESS:
        value = fn(values)
        ok = abs(value - expected) <= tol
        print(
            f"  {'OK ' if ok else 'DEV'} {label:28s} {value:12.4f}  reported {expected}"
        )
        if not ok:
            failed.append(label)
    for label, fn, bound in ROBUSTNESS_P_BELOW:
        value = fn(values)
        ok = value < bound
        print(
            f"  {'OK ' if ok else 'DEV'} {label:28s} {value:12.2e}  reported < {bound}"
        )
        if not ok:
            failed.append(label)

    configs = results["configs"]
    (t, s, p), expected = DOMINATED_CONFIG
    row = configs[
        np.isclose(configs["threshold"], t)
        & (configs["silence_ms"] == s)
        & (configs["prefix_ms"] == p)
    ].iloc[0]
    ok = not row["non_dominated"] and abs(row["p_non_dominated"] - expected) <= 0.005
    print(
        f"  {'OK ' if ok else 'DEV'} P(non-dominated) {(t, s, p)}  "
        f"{row['p_non_dominated']:.3f}, dominated={not row['non_dominated']}  "
        f"reported {expected} (dominated)"
    )
    if not ok:
        failed.append(f"P(non-dominated) {(t, s, p)}")

    tab = values["rep_by_day"]
    found = {
        int(rep): {str(day): int(n) for day, n in row.items()}
        for rep, row in tab.iterrows()
    }
    ok = found == REP_BY_DAY
    print(f"  {'OK ' if ok else 'DEV'} replication x recording date  {found}")
    if not ok:
        failed.append("replication x recording date")
    return failed


def main():
    results = analyse_sweep.run(ROOT / "results", Path(__file__).parent / "figures")
    values = robustness.run(results)
    failed = check(results) + check_robustness(results, values)
    if failed:
        sys.exit(f"\nValues deviate from the manuscript: {failed}")
    print("\nAll manuscript values reproduced.")


if __name__ == "__main__":
    main()
