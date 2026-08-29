"""
VAD Sweep Analysis
==================
Reads sweep JSONL files, fits RSM models, computes Pareto front,
and produces publication-quality PNG figures for LaTeX inclusion.

Usage:
    python3 analyse_sweep.py --input sweep_runs/ --output figures/

Configuration is at the top of this file under ANALYSIS CONFIG.

Setup (recommended):
    python3 -m venv venv
    source venv/bin/activate
    pip3 install numpy pandas scipy matplotlib scikit-learn statsmodels

Requirements:
    numpy pandas scipy matplotlib scikit-learn statsmodels
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for PNG output
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

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
    # Plot DPI — 300 is standard for IEEE publication
    "dpi": 300,
    # Figure size in inches (width, height)
    "figsize": (7, 5),
    # Colour scheme
    "color_fpr": "#2166ac",
    "color_tl": "#d6604d",
    "color_pareto": "#1a9641",
    "color_scatter": "#4d4d4d",
    "color_grid": "#cccccc",
    # Contour levels for response surface plots
    "contour_levels": 12,
}

# ── HELPERS ────────────────────────────────────────────────────────────────────


def load_jsonl(paths: list[Path]) -> pd.DataFrame:
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
    return pd.DataFrame(records)


def flatten_record(rec: dict) -> dict:
    """Flatten nested snapshot into a flat dict for analysis."""
    snap = rec.get("snapshot", {})
    openai = snap.get("openai", {})
    vad = rec.get("snapshot", {}).get("vad_config", {}) or rec.get("vad_config", {})
    ws = snap.get("websocket", {})
    inp = snap.get("input", {})

    return {
        "run_number": rec.get("run_number"),
        "design_point_index": rec.get("design_point_index"),
        "rep": rec.get("rep"),
        "session_id": rec.get("session_id"),
        "duration_s": rec.get("duration_s"),
        "annotation_valid": rec.get("annotation", {}).get("valid"),
        # Session metadata covariates — derive date from started_at directly
        "recording_date": rec.get("started_at", "")[:10] or None,
        "warmup": rec.get("annotation", {}).get("warmup", False),
        "speaker": rec.get("annotation", {}).get("speaker", 1),
        # VAD config
        "threshold": vad.get("threshold"),
        "silence_ms": vad.get("silence_duration_ms"),
        "prefix_ms": vad.get("prefix_padding_ms"),
        # Primary DVs
        "fpr": openai.get("fpr"),
        # tl_ms: response latency as reported by the API, measured from
        # speech_stopped. speech_stopped fires only AFTER server-VAD has
        # waited silence_duration_ms, so this metric excludes the end-of-turn
        # wait for the very parameter under optimisation. Kept for transparency.
        "tl_ms": openai.get("response_latency_ms"),
        # ptl_ms: user-perceived latency from the acoustic end of the utterance.
        # Adds the VAD wait back in. This is the correct optimisation target and
        # the one all Pareto/latency claims should rest on.
        "ptl_ms": (
            (vad.get("silence_duration_ms") or 0)
            + (openai.get("response_latency_ms") or 0)
        ),
        # Secondary DVs
        "ir": openai.get("interruptions"),
        "response_count": openai.get("response_count"),
        "turn_duration_mean_ms": openai.get("turn_duration_mean_ms"),
        # Covariates
        "ws_latency_ms": ws.get("latency_ms"),
        "silence_ratio": inp.get("silence_ratio"),
        "input_level_dbfs": inp.get("level_dbfs"),
    }


def apply_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    n_before = len(df)

    # Annotation filter
    f = cfg["annotation_filter"]
    if f == "valid":
        df = df[df["annotation_valid"]]
    elif f == "annotated":
        df = df[df["annotation_valid"].notna()]
    # "all" — no filter

    # Duration filter
    df = df[df["duration_s"].fillna(0) >= cfg["min_duration_s"]]

    # FPR outlier filter
    df = df[df["fpr"].fillna(1.0) <= cfg["fpr_outlier_threshold"]]

    # Exclude warm-up runs (first 2-3 runs per day)
    df = df[~df["warmup"].fillna(False)]

    # Drop rows missing primary DVs
    df = df.dropna(
        subset=["fpr", "tl_ms", "ptl_ms", "threshold", "silence_ms", "prefix_ms"]
    )

    n_after = len(df)
    print(
        f"  Runs after filtering: {n_after}/{n_before} "
        f"(removed {n_before - n_after}, filter='{f}')"
    )
    return df.reset_index(drop=True)


def code_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Convert natural units to coded [-1, 0, +1] for RSM."""
    df = df.copy()
    df["x1"] = (df["threshold"] - df["threshold"].mean()) / (
        df["threshold"].std() + 1e-9
    )
    df["x2"] = (df["silence_ms"] - df["silence_ms"].mean()) / (
        df["silence_ms"].std() + 1e-9
    )
    df["x3"] = (df["prefix_ms"] - df["prefix_ms"].mean()) / (
        df["prefix_ms"].std() + 1e-9
    )
    # Binary recording-day covariate: 0 for the earliest day, 1 for the next.
    # Entered as a linear covariate to check for between-day drift (vocal
    # fatigue, server-side inference variation, etc.). Only meaningful when
    # more than one recording date is present.
    if "recording_date" in df.columns and df["recording_date"].notna().any():
        days = sorted(df["recording_date"].dropna().unique())
        day_map = {d: float(i) for i, d in enumerate(days)}
        df["day"] = df["recording_date"].map(day_map).fillna(0.0)
    else:
        df["day"] = 0.0
    return df


def fit_rsm(
    df: pd.DataFrame, outcome: str, covariates: list[str] | None = None
) -> dict:
    """Fit second-order RSM model. Returns coefficients, R², p-values."""
    covariates = covariates or []
    X_raw = df[["x1", "x2", "x3"]].values
    y = df[outcome].values

    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X_raw)
    feature_names = poly.get_feature_names_out(["x1", "x2", "x3"])

    # Append covariates as linear terms only
    if covariates:
        active_covariates = [
            c for c in covariates if c in df.columns and df[c].std() > 1e-9
        ]
        if active_covariates:
            cov_data = df[active_covariates].fillna(0).values
            X_poly = np.hstack([X_poly, cov_data])
            feature_names = np.concatenate([feature_names, active_covariates])

        if len(active_covariates) < len(covariates):
            skipped = set(covariates) - set(active_covariates)
            print(f"  Skipping zero-variance covariates: {skipped}")

    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

    r2 = r2_score(y, y_pred)
    n, p = X_poly.shape

    # OLS t-stats and p-values
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (n - p)
    try:
        cov = mse * np.linalg.inv(X_poly.T @ X_poly)
        se = np.sqrt(np.diag(cov))
        t_stats = model.coef_ / (se + 1e-12)
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p))
    except np.linalg.LinAlgError:
        t_stats = np.full_like(model.coef_, np.nan)
        p_values = np.full_like(model.coef_, np.nan)

    return {
        "model": model,
        "poly": poly,
        "feature_names": feature_names,
        "coef": dict(zip(feature_names, model.coef_)),
        "r2": r2,
        "p_values": dict(zip(feature_names, p_values)),
        "y_pred": y_pred,
        "outcome": outcome,
    }


def pareto_front(fpr: np.ndarray, tl: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points (minimise both FPR and TL)."""
    n = len(fpr)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (
                fpr[j] <= fpr[i]
                and tl[j] <= tl[i]
                and (fpr[j] < fpr[i] or tl[j] < tl[i])
            ):
                mask[i] = False
                break
    return mask


# ── PLOTS ──────────────────────────────────────────────────────────────────────


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


def fig1_response_surfaces(
    df: pd.DataFrame, rsm_fpr: dict, rsm_tl: dict, output_dir: Path
):
    """Figure 1: Response surface plots for FPR and TL vs threshold × silence_ms."""
    fig, axes = plt.subplots(
        1, 2, figsize=(CONFIG["figsize"][0] * 1.4, CONFIG["figsize"][1])
    )

    # Grid over threshold × silence_ms at median prefix_ms
    t_range = np.linspace(df["x1"].min(), df["x1"].max(), 60)
    s_range = np.linspace(df["x2"].min(), df["x2"].max(), 60)
    T, S = np.meshgrid(t_range, s_range)
    P = np.zeros_like(T)  # prefix_padding fixed at CCD centre (coded = 0)

    X_grid = np.column_stack([T.ravel(), S.ravel(), P.ravel()])

    for ax, rsm, label, color, title in [
        (axes[0], rsm_fpr, "FPR", CONFIG["color_fpr"], "False Positive Rate"),
        (
            axes[1],
            rsm_tl,
            "Perceived latency (ms)",
            CONFIG["color_tl"],
            "Perceived latency (ms)",
        ),
    ]:
        X_poly_grid = rsm["poly"].transform(X_grid)
        # If the model was fitted with extra linear covariates (e.g. day,
        # silence_ratio) beyond the 3 design factors, hold them at 0 (their
        # coded/centred reference) so the surface is drawn at that reference.
        n_extra = rsm["model"].coef_.shape[0] - X_poly_grid.shape[1]
        if n_extra > 0:
            X_poly_grid = np.hstack(
                [X_poly_grid, np.zeros((X_poly_grid.shape[0], n_extra))]
            )
        Z = rsm["model"].predict(X_poly_grid).reshape(T.shape)

        cs = ax.contourf(
            T,
            S,
            Z,
            levels=CONFIG["contour_levels"],
            cmap="RdYlGn_r" if label == "FPR" else "RdYlGn",
            alpha=0.85,
        )
        ax.contour(
            T,
            S,
            Z,
            levels=CONFIG["contour_levels"],
            colors="white",
            linewidths=0.4,
            alpha=0.5,
        )
        plt.colorbar(cs, ax=ax, label=label, shrink=0.85)

        # Overlay observed points
        ax.scatter(
            df["x1"],
            df["x2"],
            c="white",
            s=30,
            zorder=5,
            alpha=0.9,
            linewidths=0.8,
            edgecolors=CONFIG["color_scatter"],
        )

        ax.set_xlabel("threshold (coded)")
        ax.set_ylabel("silence_duration_ms (coded)")
        ax.set_title(f"{title}\n(at median prefix_padding_ms)")

    fig.suptitle("Response Surfaces: FPR and Perceived Latency", y=1.02)
    fig.tight_layout()
    path = output_dir / "fig1_response_surfaces.png"
    fig.savefig(path, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig2_pareto_front(df: pd.DataFrame, output_dir: Path):
    """Figure 2: Pareto front in FPR–perceived-latency objective space.

    Optimisation target is ptl_ms (perceived latency = silence_duration_ms +
    measured response latency). The measured-TL front is drawn faintly for
    reference, to make the effect of the latency correction visible.
    """
    fpr = df["fpr"].values
    tl = df["ptl_ms"].values
    is_pareto = pareto_front(fpr, tl)

    # Sort Pareto points for line drawing
    pareto_df = pd.DataFrame({"fpr": fpr[is_pareto], "tl": tl[is_pareto]})
    pareto_df = pareto_df.sort_values("fpr")

    fig, ax = plt.subplots(figsize=CONFIG["figsize"])

    ax.scatter(
        fpr[~is_pareto],
        tl[~is_pareto],
        c=CONFIG["color_scatter"],
        s=20,
        alpha=0.45,
        linewidths=0,
        label="Dominated",
    )
    ax.scatter(
        fpr[is_pareto],
        tl[is_pareto],
        c=CONFIG["color_pareto"],
        s=45,
        zorder=5,
        linewidths=0.5,
        edgecolors="white",
        label="Pareto-optimal",
    )
    ax.step(
        pareto_df["fpr"],
        pareto_df["tl"],
        where="post",
        color=CONFIG["color_pareto"],
        linewidth=1.2,
        alpha=0.8,
    )

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("Perceived latency (ms)")
    ax.set_title("Pareto Front: FPR vs Perceived Latency")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "fig2_pareto_front.png"
    fig.savefig(path, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig3_rsm_coefficients(rsm_fpr: dict, rsm_tl: dict, output_dir: Path):
    """Figure 3: RSM coefficient plot with significance markers."""
    fig, axes = plt.subplots(
        1, 2, figsize=(CONFIG["figsize"][0] * 1.4, CONFIG["figsize"][1])
    )

    for ax, rsm, color, title in [
        (axes[0], rsm_fpr, CONFIG["color_fpr"], f"FPR  (R²={rsm_fpr['r2']:.3f})"),
        (
            axes[1],
            rsm_tl,
            CONFIG["color_tl"],
            f"Perceived Latency  (R²={rsm_tl['r2']:.3f})",
        ),
    ]:
        names = list(rsm["coef"].keys())
        coefs = list(rsm["coef"].values())
        pvals = [rsm["p_values"][n] for n in names]

        # Exclude intercept
        mask = [n != "1" for n in names]
        names = [n for n, m in zip(names, mask) if m]
        coefs = [c for c, m in zip(coefs, mask) if m]
        pvals = [p for p, m in zip(pvals, mask) if m]

        y_pos = range(len(names))
        colors = [color if p < 0.005 else CONFIG["color_grid"] for p in pvals]

        ax.barh(list(y_pos), coefs, color=colors, edgecolor="white", height=0.6)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Coefficient value")
        ax.set_title(title)

        # Legend
        sig_patch = mpatches.Patch(color=color, label="p < 0.005 (Bonferroni)")
        ns_patch = mpatches.Patch(color=CONFIG["color_grid"], label="n.s.")
        ax.legend(handles=[sig_patch, ns_patch], frameon=False, fontsize=7)

    fig.suptitle("RSM Coefficients (second-order model)", y=1.02)
    fig.tight_layout()
    path = output_dir / "fig3_rsm_coefficients.png"
    fig.savefig(path, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig4_marginal_effects(df: pd.DataFrame, output_dir: Path):
    """Figure 4: Marginal effect of each parameter on FPR and TL."""
    params = [
        ("threshold", "threshold"),
        ("silence_ms", "silence_duration_ms"),
        ("prefix_ms", "prefix_padding_ms"),
    ]

    fig, axes = plt.subplots(
        2, 3, figsize=(CONFIG["figsize"][0] * 1.6, CONFIG["figsize"][1] * 1.3)
    )

    for col, (col_name, label) in enumerate(params):
        for row, (outcome, color, ylabel) in enumerate(
            [
                ("fpr", CONFIG["color_fpr"], "FPR"),
                ("ptl_ms", CONFIG["color_tl"], "Perceived latency (ms)"),
            ]
        ):
            ax = axes[row][col]
            x = df[col_name].values
            y = df[outcome].values

            # Bin means
            n_bins = min(8, len(np.unique(x)))
            bins = np.linspace(x.min(), x.max(), n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_means, bin_se = [], []
            for i in range(n_bins):
                if i == n_bins - 1:
                    # Include max value in final bin
                    mask = (x >= bins[i]) & (x <= bins[i + 1])
                else:
                    mask = (x >= bins[i]) & (x < bins[i + 1])
                vals = y[mask]
                if len(vals) > 0:
                    bin_means.append(vals.mean())
                    bin_se.append(vals.std() / max(1, np.sqrt(len(vals))))
                else:
                    bin_means.append(np.nan)
                    bin_se.append(0)
            # Drop empty bins
            valid = [i for i, m in enumerate(bin_means) if not np.isnan(m)]
            bin_centers = [bin_centers[i] for i in valid]
            bin_means = [bin_means[i] for i in valid]
            bin_se = [bin_se[i] for i in valid]

            ax.scatter(x, y, c=CONFIG["color_scatter"], s=8, alpha=0.3, linewidths=0)
            ax.errorbar(
                bin_centers,
                bin_means,
                yerr=bin_se,
                fmt="o-",
                color=color,
                linewidth=1.5,
                markersize=5,
                capsize=3,
                zorder=5,
            )

            if row == 0:
                ax.set_title(label.replace("_", " ") if "_" in label else label)
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.set_xlabel(col_name.replace("_ms", " (ms)"))

    fig.suptitle("Marginal Effects on Primary Outcome Variables", y=1.02)
    fig.tight_layout()
    path = output_dir / "fig4_marginal_effects.png"
    fig.savefig(path, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig5_design_coverage(df: pd.DataFrame, output_dir: Path):
    """Figure 5: CCD design point coverage — runs completed per design point."""
    counts = df.groupby("design_point_index").size().reset_index(name="count")
    max_dp = int(df["design_point_index"].max()) + 1 if len(df) > 0 else 19

    all_idx = pd.DataFrame({"design_point_index": range(max_dp)})
    counts = all_idx.merge(counts, on="design_point_index", how="left").fillna(0)

    fig, ax = plt.subplots(figsize=CONFIG["figsize"])
    colors = [
        CONFIG["color_fpr"]
        if c >= 4
        else CONFIG["color_tl"]
        if c > 0
        else CONFIG["color_grid"]
        for c in counts["count"]
    ]
    ax.bar(
        counts["design_point_index"], counts["count"], color=colors, edgecolor="white"
    )
    ax.axhline(4, linestyle="--", color="black", linewidth=0.8, label="Target reps (4)")
    ax.set_xlabel("Design Point Index")
    ax.set_ylabel("Completed Runs")
    ax.set_title("CCD Design Coverage")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "fig5_design_coverage.png"
    fig.savefig(path, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig6_day_effect(df: pd.DataFrame, output_dir: Path):
    """Figure 6: Day-effect covariate — FPR and TL by recording day."""
    fig, axes = plt.subplots(
        1, 2, figsize=(CONFIG["figsize"][0] * 1.4, CONFIG["figsize"][1])
    )

    for ax, outcome, color, label in [
        (axes[0], "fpr", CONFIG["color_fpr"], "FPR"),
        (axes[1], "ptl_ms", CONFIG["color_tl"], "Perceived latency (ms)"),
    ]:
        days = sorted(df["recording_date"].dropna().unique())
        means = [df[df["recording_date"] == d][outcome].mean() for d in days]
        ses = [df[df["recording_date"] == d][outcome].sem() for d in days]

        ax.errorbar(
            days,
            means,
            yerr=ses,
            fmt="o-",
            color=color,
            linewidth=1.5,
            markersize=6,
            capsize=4,
        )
        ax.set_xlabel("Recording Date")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by Day\n(day-effect covariate check)")
        ax.set_xticks(range(len(days)))
        ax.set_xticklabels([str(d) for d in days], rotation=20, ha="right", fontsize=7)

    fig.suptitle("Day-Effect Covariate — Vocal Fatigue Check", y=1.02)
    fig.tight_layout()
    path = output_dir / "fig6_day_effect.png"
    fig.savefig(path, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def write_pareto_model(df: pd.DataFrame, output_dir: Path):
    """
    Write Pareto-optimal configurations to JSON for use by adaptive_tuner.py.

    Output: figures/pareto_model.json
    Schema: {"latency_metric": "perceived_latency_ms",
             "pareto_points": [{"threshold", "silence_duration_ms",
                                 "prefix_padding_ms", "fpr", "tl_ms",
                                 "perceived_latency_ms"}, ...]}

    The front is selected on perceived latency (ptl_ms = silence + measured TL),
    which is the user-relevant metric. Each point still carries the measured
    tl_ms so a consumer observing raw response_latency_ms at run time can
    reconstruct either quantity. Consumers optimising latency should read
    perceived_latency_ms (and add silence_duration_ms to any raw run-time
    response_latency_ms before comparing).
    """
    fpr = df["fpr"].values
    ptl = df["ptl_ms"].values
    is_pareto = pareto_front(fpr, ptl)

    pareto_df = df[is_pareto][
        ["threshold", "silence_ms", "prefix_ms", "fpr", "tl_ms", "ptl_ms"]
    ].sort_values("fpr")

    points = [
        {
            "threshold": float(row["threshold"]),
            "silence_duration_ms": int(row["silence_ms"]),
            "prefix_padding_ms": int(row["prefix_ms"]),
            "fpr": round(float(row["fpr"]), 6),
            "tl_ms": round(float(row["tl_ms"]), 2),
            "perceived_latency_ms": round(float(row["ptl_ms"]), 2),
        }
        for _, row in pareto_df.iterrows()
    ]

    path = output_dir / "pareto_model.json"
    path.write_text(
        json.dumps(
            {"latency_metric": "perceived_latency_ms", "pareto_points": points},
            indent=2,
        )
    )
    print(f"  Saved: {path.name}  ({len(points)} Pareto-optimal point(s))")


def write_summary(
    df: pd.DataFrame,
    rsm_fpr: dict,
    rsm_tl: dict,
    output_dir: Path,
    rsm_ptl: dict | None = None,
):
    """Write a plain-text summary table for quick inspection."""
    lines = [
        "VAD SWEEP ANALYSIS SUMMARY",
        "=" * 50,
        f"Runs included:  {len(df)}",
        f"Annotation filter: {CONFIG['annotation_filter']}",
        "",
        "Primary outcome descriptives:",
        (
            f"  FPR              mean={df['fpr'].mean():.3f}  sd={df['fpr'].std():.3f}  "
            f"min={df['fpr'].min():.3f}  max={df['fpr'].max():.3f}"
        ),
        (
            f"  TL (measured)    mean={df['tl_ms'].mean():.1f}ms  sd={df['tl_ms'].std():.1f}  "
            f"min={df['tl_ms'].min():.1f}  max={df['tl_ms'].max():.1f}"
        ),
        (
            f"  Perceived latency mean={df['ptl_ms'].mean():.1f}ms  sd={df['ptl_ms'].std():.1f}  "
            f"min={df['ptl_ms'].min():.1f}  max={df['ptl_ms'].max():.1f}"
        ),
        "",
        "  NOTE: measured TL is taken from speech_stopped, which fires only after",
        "  server-VAD has already waited silence_duration_ms. Perceived latency adds",
        "  that wait back in and is the metric all latency claims rest on.",
        "",
        "RSM model fit:",
        f"  FPR model R²                = {rsm_fpr['r2']:.4f}",
        f"  TL (measured) model R²      = {rsm_tl['r2']:.4f}",
    ]
    if rsm_ptl is not None:
        lines.append(f"  Perceived latency model R²  = {rsm_ptl['r2']:.4f}")
    lines += [
        "",
        "Significant RSM terms (p < 0.005, Bonferroni):",
    ]

    rsm_iter = [(rsm_fpr, "FPR"), (rsm_tl, "TL(measured)")]
    if rsm_ptl is not None:
        rsm_iter.append((rsm_ptl, "PerceivedLatency"))
    for rsm, label in rsm_iter:
        sig = [k for k, v in rsm["p_values"].items() if v < 0.005 and k != "1"]
        lines.append(f"  {label}: {', '.join(sig) if sig else 'none'}")

    fpr = df["fpr"].values
    ptl = df["ptl_ms"].values
    is_pareto = pareto_front(fpr, ptl)
    lines += [
        "",
        f"Pareto-optimal runs (FPR vs perceived latency): {is_pareto.sum()} of {len(df)}",
    ]

    pareto_df = df[is_pareto][
        ["threshold", "silence_ms", "prefix_ms", "fpr", "tl_ms", "ptl_ms"]
    ].sort_values("fpr")
    lines.append("")
    lines.append("Pareto front configurations:")
    lines.append(
        f"  {'threshold':>10} {'silence_ms':>12} {'prefix_ms':>10} "
        f"{'FPR':>8} {'TL_meas':>9} {'Perceived':>10}"
    )
    for _, row in pareto_df.iterrows():
        lines.append(
            f"  {row['threshold']:>10.3f} {row['silence_ms']:>12.0f} "
            f"{row['prefix_ms']:>10.0f} {row['fpr']:>8.4f} {row['tl_ms']:>9.1f} "
            f"{row['ptl_ms']:>10.1f}"
        )

    path = output_dir / "summary.txt"
    path.write_text("\n".join(lines))
    print(f"  Saved: {path.name}")
    print()
    print("\n".join(lines[:20]))


# ── MAIN ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="VAD sweep analysis")
    parser.add_argument(
        "--input", default="sweep_runs", help="Directory containing sweep_*.jsonl files"
    )
    parser.add_argument(
        "--output", default="figures", help="Output directory for PNG figures"
    )
    parser.add_argument(
        "--filter",
        choices=["all", "valid", "annotated"],
        default=None,
        help="Override annotation_filter in CONFIG",
    )
    args = parser.parse_args()

    if args.filter:
        CONFIG["annotation_filter"] = args.filter

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    jsonl_files = sorted(input_dir.glob("sweep_*.jsonl"))
    if not jsonl_files:
        print(f"No sweep_*.jsonl files found in {input_dir}")
        sys.exit(1)

    print(f"Loading {len(jsonl_files)} file(s) from {input_dir}")
    raw_df = load_jsonl(jsonl_files)
    print(f"  Raw records: {len(raw_df)}")

    # Flatten
    flat = [flatten_record(r) for r in raw_df.to_dict("records")]
    df = pd.DataFrame(flat)

    # Filter
    df = apply_filters(df, CONFIG)
    if len(df) < 5:
        print(f"Too few runs after filtering ({len(df)}). Adjust CONFIG filters.")
        sys.exit(1)

    # Code variables
    df = code_variables(df)

    # Fit RSM
    print("\nFitting RSM models...")
    rsm_fpr = fit_rsm(df, "fpr", covariates=["silence_ratio", "day"])
    rsm_tl = fit_rsm(df, "tl_ms", covariates=["silence_ratio", "day"])
    rsm_ptl = fit_rsm(df, "ptl_ms", covariates=["silence_ratio", "day"])
    print(f"  FPR R²               = {rsm_fpr['r2']:.4f}")
    print(f"  TL (measured) R²     = {rsm_tl['r2']:.4f}")
    print(f"  Perceived latency R² = {rsm_ptl['r2']:.4f}")

    # Plots
    print(f"\nGenerating figures -> {output_dir}/")
    set_style()
    fig1_response_surfaces(df, rsm_fpr, rsm_ptl, output_dir)
    fig2_pareto_front(df, output_dir)
    write_pareto_model(df, output_dir)
    fig3_rsm_coefficients(rsm_fpr, rsm_ptl, output_dir)
    fig4_marginal_effects(df, output_dir)
    write_summary(df, rsm_fpr, rsm_tl, output_dir, rsm_ptl=rsm_ptl)

    print(
        f"\nDone. {len(list(output_dir.glob('*.png')))} figures written to {output_dir}/"
    )


if __name__ == "__main__":
    main()
