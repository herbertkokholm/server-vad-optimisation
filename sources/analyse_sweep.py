"""
VAD Sweep Analysis
==================
Reads sweep JSONL files, fits second-order RSM models, computes the Pareto
front of configuration means with bootstrap uncertainty, and produces
journal-ready figures (EPS, 600 dpi TIFF and PNG) plus pareto_model.json.

Usage:
    python3 analyse_sweep.py --input sweep_runs/ --output figures/

Configuration is at the top of this file under ANALYSIS CONFIG.

Outcomes:
    FPR  interruptive false-positive rate (snapshot.openai.fpr)
    TL   endpointing latency (ms) = silence_duration_ms + latency from
         speech_stopped to the first response audio chunk. speech_stopped
         only fires after server VAD has waited silence_duration_ms, so the
         API-reported latency alone excludes the parameter being optimised.

Model:
    y = b0 + sum b_i x_i + sum_{i<j} b_ij x_i x_j + sum b_ii x_i^2 + b_day day
    fitted by OLS with each VAD parameter z-scored (sample SD) and day a
    binary recording-day indicator. Terms are tested at a Bonferroni-corrected
    alpha of 0.05 / 10.

Setup (recommended):
    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from PIL import Image
from scipy import stats

# ── ANALYSIS CONFIG ────────────────────────────────────────────────────────────
# Edit these values to control filtering and output behaviour.

CONFIG = {
    # Annotation filter:
    #   "all"       — include all runs regardless of annotation.valid
    #   "valid"     — include only runs where annotation.valid == True
    #   "annotated" — include only runs where annotation.valid is not None
    "annotation_filter": "valid",
    # Minimum session duration in seconds — exclude runs shorter than this
    # (e.g. failed sessions, operator_did_not_pick_up)
    "min_duration_s": 60,
    # FPR outlier threshold — exclude runs above this value (0–1)
    # Set to 1.0 to include all
    "fpr_outlier_threshold": 1.0,
    # Bootstrap resamples and random seed (figures and Pareto frequencies)
    "n_boot": 2000,
    "seed": 20260423,
    # Bonferroni-corrected significance level: 9 RSM terms + day
    "alpha": 0.05 / 10,
    # Raster output resolution (TIFF and PNG)
    "dpi": 600,
    # Figure width: 174 mm = full page width (Springer Nature)
    "fig_width_mm": 174,
}

FACTORS = ["threshold", "silence_ms", "prefix_ms"]
AXIS_LABEL = {
    "threshold": "Activation threshold",
    "silence_ms": "Silence duration (ms)",
    "prefix_ms": "Prefix padding (ms)",
    "fpr": "Interruptive false-positive rate (FPR)",
    "tl_ms": "Endpointing latency, TL (ms)",
}
RSM_FORMULA = "x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3 + I(x1**2) + I(x2**2) + I(x3**2)"
RSM_TERMS = [
    ("x1", "Threshold"),
    ("x2", "Silence"),
    ("x3", "Prefix"),
    ("x1:x2", "Threshold × Silence"),
    ("x1:x3", "Threshold × Prefix"),
    ("x2:x3", "Silence × Prefix"),
    ("I(x1 ** 2)", "Threshold²"),
    ("I(x2 ** 2)", "Silence²"),
    ("I(x3 ** 2)", "Prefix²"),
    ("day", "Recording day"),
]

# Okabe-Ito colours. Silence levels use orange / blue / black, which differ in
# luminance and stay distinguishable in greyscale.
OI_ORANGE, OI_BLUE, OI_VERMILION, OI_BLACK = "#E69F00", "#0072B2", "#D55E00", "#000000"
SILENCE_COLOURS = [OI_ORANGE, OI_BLUE, OI_BLACK]
THRESHOLD_MARKERS = ["o", "s", "^"]
SESSION_GREY = "#BDBDBD"  # data points only, never used for text
MM = 1 / 25.4

# ── DATA ───────────────────────────────────────────────────────────────────────


def load_jsonl(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  Warning: skipping malformed line in {path.name}: {e}")
    return records


def flatten_record(rec: dict) -> dict:
    """Flatten nested snapshot into a flat dict for analysis."""
    snap = rec.get("snapshot", {})
    openai = snap.get("openai", {})
    vad = snap.get("vad_config", {}) or rec.get("vad_config", {})
    ann = rec.get("annotation", {})
    silence_ms = vad.get("silence_duration_ms")
    response_latency_ms = openai.get("response_latency_ms")

    return {
        "run_number": rec.get("run_number"),
        "design_point": rec.get("design_point_index"),
        "rep": rec.get("rep"),
        "session_id": rec.get("session_id"),
        "duration_s": rec.get("duration_s"),
        "annotation_valid": ann.get("valid"),
        "warmup": ann.get("warmup", False),
        "speaker": ann.get("speaker", 1),
        "recording_date": rec.get("started_at", "")[:10] or None,
        # VAD config
        "threshold": vad.get("threshold"),
        "silence_ms": silence_ms,
        "prefix_ms": vad.get("prefix_padding_ms"),
        # Primary DVs
        "fpr": openai.get("fpr"),
        "vad_triggered": openai.get("vad_triggered"),
        "vad_false_triggers": openai.get("vad_false_triggers"),
        "interruptions": openai.get("interruptions"),
        # speech_stopped -> first response audio chunk, as measured on the
        # event stream. Excludes the silence_duration_ms wait.
        "response_latency_ms": response_latency_ms,
        # TL: endpointing latency from the acoustic end of the utterance
        "tl_ms": (
            silence_ms + response_latency_ms
            if silence_ms is not None and response_latency_ms is not None
            else None
        ),
        "ws_latency_ms": snap.get("websocket", {}).get("latency_ms"),
    }


def apply_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    n_before = len(df)

    # Annotation filter
    f = cfg["annotation_filter"]
    if f == "valid":
        df = df[df["annotation_valid"] == True]
    elif f == "annotated":
        df = df[df["annotation_valid"].notna()]
    # "all" — no filter

    # Duration filter
    df = df[df["duration_s"].fillna(0) >= cfg["min_duration_s"]]

    # FPR outlier filter
    df = df[df["fpr"].fillna(1.0) <= cfg["fpr_outlier_threshold"]]

    # Exclude warm-up runs (first 2-3 runs per day)
    df = df[~df["warmup"].fillna(False).astype(bool)]

    # Drop rows missing primary DVs
    df = df.dropna(subset=["fpr", "tl_ms", *FACTORS])

    n_after = len(df)
    print(
        f"  Runs after filtering: {n_after}/{n_before} "
        f"(removed {n_before - n_after}, filter='{f}')"
    )
    return df.reset_index(drop=True)


def code_variables(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Z-score the VAD parameters (sample SD) and add the recording-day indicator.

    Returns the coded frame and {factor: (mean, sd)} for back-transformation.
    """
    df = df.copy()
    scaling = {}
    for i, f in enumerate(FACTORS, start=1):
        mean, sd = df[f].mean(), df[f].std(ddof=1)
        scaling[f] = (mean, sd)
        df[f"x{i}"] = (df[f] - mean) / sd
    # Recording day: 0 for the earliest date, 1 for the next. Entered as a
    # linear covariate to check for between-day drift (vocal fatigue,
    # server-side inference variation, etc.).
    days = sorted(df["recording_date"].dropna().unique())
    df["day"] = df["recording_date"].map({d: float(i) for i, d in enumerate(days)})
    df["day"] = df["day"].fillna(0.0)
    return df, scaling


def fit_rsm(df: pd.DataFrame, outcome: str, with_day: bool = True):
    """Fit the second-order RSM by OLS. Day is dropped if only one day exists."""
    formula = f"{outcome} ~ {RSM_FORMULA}"
    if with_day and df["day"].nunique() > 1:
        formula += " + day"
    return smf.ols(formula, data=df).fit()


def levels(df: pd.DataFrame, factor: str) -> list:
    return sorted(df[factor].unique())


def boot_ci(
    values: np.ndarray, rng: np.random.Generator, n_boot: int
) -> tuple[float, float]:
    """Percentile 95% CI of the mean."""
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    return tuple(np.percentile(values[idx].mean(axis=1), [2.5, 97.5]))


def pareto_front(fpr: np.ndarray, tl: np.ndarray) -> np.ndarray:
    """Return boolean mask of non-dominated points (minimise both FPR and TL)."""
    le = (fpr[None, :] <= fpr[:, None]) & (tl[None, :] <= tl[:, None])
    lt = (fpr[None, :] < fpr[:, None]) | (tl[None, :] < tl[:, None])
    return ~(le & lt).any(axis=1)


def config_table(
    df: pd.DataFrame, rng: np.random.Generator, n_boot: int
) -> pd.DataFrame:
    """Configuration means with bootstrap CIs and non-dominance frequencies.

    Sessions are pooled per unique VAD configuration, so the five replicated
    centre-point blocks of the CCD form a single configuration (n = 20) and
    the face-centred design yields 15 configurations.
    """
    groups = dict(list(df.groupby(FACTORS)))
    dp = df.groupby(FACTORS, as_index=False).agg(
        threshold=("threshold", "first"),
        silence_ms=("silence_ms", "first"),
        prefix_ms=("prefix_ms", "first"),
        n=("fpr", "size"),
        fpr=("fpr", "mean"),
        tl_ms=("tl_ms", "mean"),
        response_latency_ms=("response_latency_ms", "mean"),
    )
    for i, key in enumerate(groups):
        g = groups[key]
        dp.loc[i, ["fpr_lo", "fpr_hi"]] = boot_ci(g["fpr"].to_numpy(), rng, n_boot)
        dp.loc[i, ["tl_lo", "tl_hi"]] = boot_ci(g["tl_ms"].to_numpy(), rng, n_boot)
    dp["non_dominated"] = pareto_front(dp["fpr"].to_numpy(), dp["tl_ms"].to_numpy())

    # How often is each configuration non-dominated when its sessions are
    # resampled with replacement?
    counts = np.zeros(len(dp))
    samples = [(g["fpr"].to_numpy(), g["tl_ms"].to_numpy()) for g in groups.values()]
    for _ in range(n_boot):
        fpr_b, tl_b = [], []
        for fpr, tl in samples:
            idx = rng.integers(0, len(fpr), len(fpr))
            fpr_b.append(fpr[idx].mean())
            tl_b.append(tl[idx].mean())
        counts += pareto_front(np.array(fpr_b), np.array(tl_b))
    dp["p_non_dominated"] = counts / n_boot
    return dp


# ── PLOTS ──────────────────────────────────────────────────────────────────────


def set_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 0.8,
            "patch.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            # Embed fonts as TrueType (Type 42) in EPS/PDF
            "ps.fonttype": 42,
            "pdf.fonttype": 42,
        }
    )


def figure(height_mm: float, **kwargs):
    width = CONFIG["fig_width_mm"] * MM
    return plt.subplots(figsize=(width, height_mm * MM), layout="constrained", **kwargs)


def panel_label(ax, letter: str):
    ax.text(
        -0.02,
        1.02,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def save(fig, output_dir: Path, name: str):
    """Write <name>.eps (vector), <name>.tif (RGB, LZW) and <name>.png."""
    dpi = CONFIG["dpi"]
    # Fixed EPS timestamp so regenerated figures are byte-identical
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    fig.savefig(output_dir / f"{name}.eps", format="eps")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    img = Image.open(buf).convert("RGB")
    img.save(output_dir / f"{name}.png", dpi=(dpi, dpi))
    img.save(output_dir / f"{name}.tif", dpi=(dpi, dpi), compression="tiff_lzw")
    plt.close(fig)
    print(f"  Saved: {name}.eps/.tif/.png")


def fig1_response_surfaces(df, scaling, models, output_dir: Path):
    """Figure 1: Model-predicted FPR and TL over threshold × silence duration.

    Prefix padding is fixed at its centre level; surfaces are evaluated at the
    mean of the day indicator, i.e. averaged over recording days.
    """
    fig, axes = figure(70, ncols=2)
    thr_levels, sil_levels = levels(df, "threshold"), levels(df, "silence_ms")
    prefix_centre = levels(df, "prefix_ms")[len(levels(df, "prefix_ms")) // 2]
    T, S = np.meshgrid(
        np.linspace(thr_levels[0], thr_levels[-1], 200),
        np.linspace(sil_levels[0], sil_levels[-1], 200),
    )
    grid = pd.DataFrame(
        {
            "x1": ((T - scaling["threshold"][0]) / scaling["threshold"][1]).ravel(),
            "x2": ((S - scaling["silence_ms"][0]) / scaling["silence_ms"][1]).ravel(),
            "x3": (prefix_centre - scaling["prefix_ms"][0]) / scaling["prefix_ms"][1],
            "day": df["day"].mean(),
        }
    )
    design = df[["threshold", "silence_ms"]].drop_duplicates()
    # Colormap: viridis_r in both panels, so darker = higher FPR / longer TL = worse
    for ax, key, letter in zip(axes, ["fpr", "tl_ms"], "ab"):
        Z = models[key].predict(grid).to_numpy().reshape(T.shape)
        cf = ax.contourf(T, S, Z, levels=12, cmap="viridis_r")
        ax.contour(T, S, Z, levels=cf.levels, colors="white", linewidths=0.3)
        for edge, lw, z in [("white", 1.6, 3), ("black", 0.6, 4)]:
            ax.scatter(
                design["threshold"],
                design["silence_ms"],
                s=22,
                facecolors="none",
                edgecolors=edge,
                linewidths=lw,
                zorder=z,
                clip_on=False,
            )
        ax.set_xlabel(AXIS_LABEL["threshold"])
        ax.set_ylabel(AXIS_LABEL["silence_ms"])
        ax.set_xticks(thr_levels)
        ax.set_yticks(sil_levels)
        # Tick marks would cut through the design-point circles on the frame
        ax.tick_params(length=0, pad=4)
        ax.spines[["top", "right"]].set_visible(True)
        cb = fig.colorbar(cf, ax=ax, shrink=0.95)
        cb.set_label(AXIS_LABEL[key])
        cb.outline.set_linewidth(0.6)
        cb.ax.tick_params(width=0.6)
        panel_label(ax, letter)
    save(fig, output_dir, "Fig1")


def fig2_pareto(df, dp, session_front, output_dir: Path):
    """Figure 2: FPR vs TL — sessions, configuration means with 95% bootstrap
    CIs, the non-dominated set of means, and the session-level front."""
    fig, ax = figure(105)
    colour = dict(zip(levels(df, "silence_ms"), SILENCE_COLOURS))
    marker = dict(zip(levels(df, "threshold"), THRESHOLD_MARKERS))

    ax.scatter(df["fpr"], df["tl_ms"], s=7, color=SESSION_GREY, linewidths=0)
    for _, r in dp.iterrows():
        c = colour[r["silence_ms"]]
        ax.errorbar(
            r["fpr"],
            r["tl_ms"],
            xerr=[[r["fpr"] - r["fpr_lo"]], [r["fpr_hi"] - r["fpr"]]],
            yerr=[[r["tl_ms"] - r["tl_lo"]], [r["tl_hi"] - r["tl_ms"]]],
            fmt="none",
            ecolor=c,
            elinewidth=0.6,
            capsize=1.5,
            capthick=0.6,
            zorder=2,
        )
        if r["non_dominated"]:
            # Outer ring with a white gap, visible on every fill colour
            ax.scatter(
                r["fpr"],
                r["tl_ms"],
                marker=marker[r["threshold"]],
                s=110,
                facecolor="white",
                edgecolor="black",
                linewidths=1.0,
                zorder=4,
            )
        ax.scatter(
            r["fpr"],
            r["tl_ms"],
            marker=marker[r["threshold"]],
            s=34,
            facecolor=c,
            edgecolor="black",
            linewidths=0.5,
            zorder=5,
        )

    front = dp[dp["non_dominated"]].sort_values("fpr")
    ax.plot(
        front["fpr"],
        front["tl_ms"],
        drawstyle="steps-post",
        color="black",
        lw=0.9,
        ls="--",
        zorder=3,
    )
    ax.scatter(
        session_front["fpr"],
        session_front["tl_ms"],
        s=80,
        facecolors="none",
        edgecolors=OI_VERMILION,
        linewidths=1.0,
        zorder=5,
    )
    ax.set_xlabel(AXIS_LABEL["fpr"])
    ax.set_ylabel(AXIS_LABEL["tl_ms"])
    ax.set_xlim(left=-0.015)

    silence_handles = [
        Line2D([], [], ls="", marker="s", ms=6, mfc=c, mec="black", mew=0.5)
        for c in colour.values()
    ]
    threshold_handles = [
        Line2D([], [], ls="", marker=m, ms=5.5, mfc="white", mec="black", mew=0.6)
        for m in marker.values()
    ]
    other_handles = [
        Line2D([], [], ls="", marker="o", ms=3, mfc=SESSION_GREY, mec="none"),
        Line2D(
            [],
            [],
            ls="--",
            lw=0.9,
            color="black",
            marker="o",
            ms=8.5,
            mfc="white",
            mec="black",
            mew=1.0,
        ),
        Line2D([], [], ls="", marker="o", ms=8, mfc="none", mec=OI_VERMILION, mew=1.0),
    ]
    legends = [
        (
            silence_handles,
            [f"{s:g} ms" for s in colour],
            "Silence duration",
            1.0,
        ),
        (
            threshold_handles,
            [f"{t:g}" for t in marker],
            "Activation threshold",
            0.68,
        ),
        (
            other_handles,
            ["Session", "Non-dominated means", "Non-dominated sessions"],
            None,
            0.36,
        ),
    ]
    # Earlier legends are kept as extra artists; the last one stays the axes
    # legend so constrained layout reserves space for the legend column.
    for i, (handles, labels, title, y) in enumerate(legends):
        leg = ax.legend(
            handles,
            labels,
            title=title,
            loc="upper left",
            bbox_to_anchor=(1.01, y),
            alignment="left",
        )
        if i < len(legends) - 1:
            ax.add_artist(leg)
    save(fig, output_dir, "Fig2")


def fig3_rsm_coefficients(models, output_dir: Path):
    """Figure 3: Standardised RSM coefficients with 95% CIs.

    Filled bars = significant at the Bonferroni level; open bars = not.
    """
    fig, axes = figure(75, ncols=2)
    specs = [
        ("fpr", OI_BLUE, "Coefficient (FPR per SD)"),
        ("tl_ms", OI_VERMILION, "Coefficient (ms per SD)"),
    ]
    for ax, (key, colour, xlabel), letter in zip(axes, specs, "ab"):
        m = models[key]
        terms = [(t, label) for t, label in RSM_TERMS if t in m.params]
        ypos = np.arange(len(terms))[::-1]
        ci = m.conf_int()
        for y, (term, _) in zip(ypos, terms):
            sig = m.pvalues[term] < CONFIG["alpha"]
            ax.barh(
                y,
                m.params[term],
                height=0.6,
                color=colour if sig else "white",
                edgecolor=colour,
                linewidth=0.8,
                zorder=2,
            )
            lo, hi = ci.loc[term]
            ax.plot([lo, hi], [y, y], color="black", lw=0.7, zorder=3)
            for x in (lo, hi):
                ax.plot([x, x], [y - 0.12, y + 0.12], color="black", lw=0.7, zorder=3)
        ax.axvline(0, color="black", lw=0.6, zorder=1)
        ax.set_yticks(ypos, [label for _, label in terms])
        ax.set_xlabel(xlabel)
        ax.text(
            1.0,
            1.02,
            f"R² = {m.rsquared:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )
        panel_label(ax, letter)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="black", edgecolor="black", lw=0.8),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", lw=0.8),
        Line2D([], [], color="black", lw=0.7),
    ]
    labels = [
        f"p < {CONFIG['alpha']:g} (Bonferroni)",
        "Not significant",
        "95% confidence interval",
    ]
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    save(fig, output_dir, "Fig3")


def fig4_marginal_effects(df, rng: np.random.Generator, output_dir: Path):
    """Figure 4: Marginal effects — sessions and level means with 95% bootstrap
    CIs for each factor (columns) and outcome (rows)."""
    fig, axes = figure(110, nrows=2, ncols=3, sharey="row")
    rows = [("fpr", OI_BLUE, "o"), ("tl_ms", OI_VERMILION, "s")]
    for r, (key, colour, marker) in enumerate(rows):
        for c, f in enumerate(FACTORS):
            ax = axes[r, c]
            lv = levels(df, f)
            jitter = 0.04 * (lv[-1] - lv[0])
            x = df[f] + rng.uniform(-jitter, jitter, len(df))
            ax.scatter(x, df[key], s=5, color=SESSION_GREY, linewidths=0, zorder=1)
            means, err_lo, err_hi = [], [], []
            for level in lv:
                v = df.loc[df[f] == level, key].to_numpy()
                lo, hi = boot_ci(v, rng, CONFIG["n_boot"])
                means.append(v.mean())
                err_lo.append(v.mean() - lo)
                err_hi.append(hi - v.mean())
            ax.errorbar(
                lv,
                means,
                yerr=[err_lo, err_hi],
                color=colour,
                marker=marker,
                ms=4.5,
                mec="black",
                mew=0.5,
                lw=0.9,
                elinewidth=0.8,
                capsize=2,
                capthick=0.8,
                zorder=3,
            )
            ax.set_xticks(lv, [f"{level:g}" for level in lv])
            ax.set_xlabel(AXIS_LABEL[f])
            if c == 0:
                ax.set_ylabel(AXIS_LABEL[key].replace("rate ", "rate\n"))
            panel_label(ax, "abcdef"[r * 3 + c])
    save(fig, output_dir, "Fig4")


# ── OUTPUT FILES ───────────────────────────────────────────────────────────────


def write_pareto_model(cfg: pd.DataFrame, output_dir: Path):
    """
    Write the non-dominated configuration means for use by adaptive_tuner.py.

    Output: <output>/pareto_model.json
    Schema: {"latency_metric": "endpointing_latency_ms",
             "selection": "configuration means",
             "pareto_points": [{"threshold", "silence_duration_ms",
                                "prefix_padding_ms", "fpr",
                                "endpointing_latency_ms", "response_latency_ms",
                                "p_non_dominated", "n_sessions"}, ...]}

    Sessions are pooled per VAD configuration, so the replicated centre point
    is represented by all its sessions rather than by its best block.
    """
    front = cfg[cfg["non_dominated"]].sort_values("fpr")
    points = [
        {
            "threshold": float(r["threshold"]),
            "silence_duration_ms": int(r["silence_ms"]),
            "prefix_padding_ms": int(r["prefix_ms"]),
            "fpr": round(float(r["fpr"]), 4),
            "endpointing_latency_ms": round(float(r["tl_ms"]), 1),
            "response_latency_ms": round(float(r["response_latency_ms"]), 1),
            "p_non_dominated": round(float(r["p_non_dominated"]), 3),
            "n_sessions": int(r["n"]),
        }
        for _, r in front.iterrows()
    ]
    path = output_dir / "pareto_model.json"
    path.write_text(
        json.dumps(
            {
                "latency_metric": "endpointing_latency_ms",
                "selection": "configuration means",
                "pareto_points": points,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  Saved: {path.name}")


def coefficient_lines(m) -> list[str]:
    ci = m.conf_int()
    lines = [f"    {'term':<22}{'coef':>10}{'95% CI':>22}{'p':>9}"]
    for term, label in [("Intercept", "Intercept"), *RSM_TERMS]:
        if term not in m.params:
            continue
        sig = " *" if term != "Intercept" and m.pvalues[term] < CONFIG["alpha"] else ""
        lo, hi = ci.loc[term]
        lines.append(
            f"    {label:<22}{m.params[term]:>10.4f}"
            f"   [{lo:>8.4f}, {hi:>8.4f}]{m.pvalues[term]:>9.4f}{sig}"
        )
    return lines


def write_summary(
    df, scaling, models, models_noday, configs, session_front, output_dir
):
    lines = [
        "VAD SWEEP ANALYSIS SUMMARY",
        "=" * 60,
        (
            f"Sessions included: {len(df)}  (annotation filter: "
            f"{CONFIG['annotation_filter']}, warm-up excluded)"
        ),
        f"Design points: {df['design_point'].nunique()}",
        f"Recording days: {df['recording_date'].value_counts().sort_index().to_dict()}",
        "",
        "Outcome descriptives (SD with ddof = 1):",
    ]
    for key, label in [
        ("fpr", "FPR"),
        ("tl_ms", "TL (endpointing, ms)"),
        ("response_latency_ms", "Latency from speech_stopped (ms)"),
    ]:
        s = df[key]
        lines.append(
            f"  {label:<34} mean={s.mean():.3f}  sd={s.std():.3f}  "
            f"min={s.min():.3f}  max={s.max():.3f}"
        )
    lines += [
        "",
        "  TL = silence_duration_ms + latency from speech_stopped. speech_stopped",
        "  fires only after server VAD has waited silence_duration_ms, so the",
        "  API-reported latency alone excludes the parameter being optimised.",
        "",
        "Predictor standardisation (z-score, sample SD):",
    ]
    for f in FACTORS:
        lines.append(f"  {f:<12} mean={scaling[f][0]:.3f}  sd={scaling[f][1]:.3f}")

    for key, label in [
        ("fpr", "FPR"),
        ("tl_ms", "TL"),
        ("response_latency_ms", "Latency from speech_stopped"),
    ]:
        m, m0 = models[key], models_noday[key]
        lines += [
            "",
            (
                f"{label} model: R² = {m.rsquared:.4f} with day, "
                f"{m0.rsquared:.4f} without day   (* p < {CONFIG['alpha']:g})"
            ),
            "  with day:",
            *coefficient_lines(m),
            "  without day:",
            *coefficient_lines(m0),
        ]

    # Secondary outcomes (descriptive analysis)
    ir = df["interruptions"] / (df["duration_s"] / 60)  # cancellations per minute
    r_ir = stats.pearsonr(df["silence_ms"], ir)
    ws = df["ws_latency_ms"]
    lines += [
        "",
        "Secondary outcomes:",
        (
            f"  IR (cancellations/min)  mean={ir.mean():.3f}  sd={ir.std():.3f}  "
            f"r(silence)={r_ir.statistic:.3f}  p={r_ir.pvalue:.2g}"
        ),
        (
            f"  WebSocket RTT (ms)      mean={ws.mean():.1f}  sd={ws.std():.1f}  "
            f"min={ws.min():.1f}  max={ws.max():.1f}"
        ),
    ]

    lines += [
        "",
        (
            "Configuration means (sessions pooled per configuration) and bootstrap "
            f"non-dominance frequency ({CONFIG['n_boot']} resamples); the "
            "non-dominated rows are written to pareto_model.json:"
        ),
        configs[[*FACTORS, "n", "fpr", "tl_ms", "non_dominated", "p_non_dominated"]]
        .sort_values("p_non_dominated", ascending=False)
        .round({"fpr": 3, "tl_ms": 1, "p_non_dominated": 3})
        .to_string(index=False),
        "",
        "Session-level non-dominated runs (single replications):",
        session_front[["design_point", "rep", *FACTORS, "fpr", "tl_ms"]]
        .round({"fpr": 4, "tl_ms": 1})
        .to_string(index=False),
    ]

    path = output_dir / "summary.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {path.name}")


# ── PIPELINE ───────────────────────────────────────────────────────────────────


def run(input_dir: Path, output_dir: Path) -> dict:
    """Run the full analysis and write all outputs. Returns intermediate results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(input_dir.glob("sweep_*.jsonl"))
    if not jsonl_files:
        print(f"No sweep_*.jsonl files found in {input_dir}")
        sys.exit(1)
    print(f"Loading {len(jsonl_files)} file(s) from {input_dir}")
    records = load_jsonl(jsonl_files)
    print(f"  Raw records: {len(records)}")

    df = apply_filters(pd.DataFrame([flatten_record(r) for r in records]), CONFIG)
    if len(df) < 12:
        print(f"Too few runs after filtering ({len(df)}). Adjust CONFIG filters.")
        sys.exit(1)
    df, scaling = code_variables(df)

    print("\nFitting RSM models...")
    outcomes = ["fpr", "tl_ms", "response_latency_ms"]
    models = {k: fit_rsm(df, k) for k in outcomes}
    models_noday = {k: fit_rsm(df, k, with_day=False) for k in outcomes}
    print(f"  FPR R² = {models['fpr'].rsquared:.4f}")
    print(f"  TL  R² = {models['tl_ms'].rsquared:.4f}")

    # Each random step gets its own stream so results do not depend on the
    # order in which outputs are produced.
    seed = CONFIG["seed"]
    configs = config_table(df, np.random.default_rng([seed, 1]), CONFIG["n_boot"])
    session_front = df[pareto_front(df["fpr"].to_numpy(), df["tl_ms"].to_numpy())]

    print(f"\nWriting outputs -> {output_dir}/")
    set_style()
    fig1_response_surfaces(df, scaling, models, output_dir)
    fig2_pareto(df, configs, session_front, output_dir)
    fig3_rsm_coefficients(models, output_dir)
    fig4_marginal_effects(df, np.random.default_rng([seed, 2]), output_dir)
    write_pareto_model(configs, output_dir)
    write_summary(df, scaling, models, models_noday, configs, session_front, output_dir)

    return {
        "df": df,
        "scaling": scaling,
        "models": models,
        "models_noday": models_noday,
        "configs": configs,
        "session_front": session_front,
    }


def main():
    parser = argparse.ArgumentParser(description="VAD sweep analysis")
    parser.add_argument(
        "--input", default="sweep_runs", help="Directory containing sweep_*.jsonl files"
    )
    parser.add_argument("--output", default="figures", help="Output directory")
    parser.add_argument(
        "--filter",
        choices=["all", "valid", "annotated"],
        default=None,
        help="Override annotation_filter in CONFIG",
    )
    args = parser.parse_args()

    if args.filter:
        CONFIG["annotation_filter"] = args.filter

    run(Path(args.input), Path(args.output))
    print("\nDone.")


if __name__ == "__main__":
    main()
