"""
Robustness analyses for Reviewer 1: replicate dependence and binomial FPR.

Writes papers/paper1/figures/robustness.txt (Appendix A): a mixed model with a
random intercept per design point, OLS on the 19 design-point means, a
lack-of-fit test against pure error, and a binomial GLM for FPR on trigger
counts.

Usage (from the repository root):
    python3 papers/paper1/robustness.py

make_figures.py calls run() with the results of its own pipeline run, so one
command regenerates everything. Run standalone, the pipeline outputs go to a
temporary directory and no other committed file is touched.
"""

import contextlib
import io
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).parent / "figures" / "robustness.txt"
sys.path.insert(0, str(ROOT / "sources"))

import analyse_sweep

TERMS = [
    "x1",
    "x2",
    "x3",
    "x1:x2",
    "x1:x3",
    "x2:x3",
    "I(x1 ** 2)",
    "I(x2 ** 2)",
    "I(x3 ** 2)",
]
NAMES = dict(
    zip(
        TERMS,
        ["Threshold", "Silence", "Prefix", "Thr×Sil", "Thr×Pre", "Sil×Pre"]
        + ["Thr²", "Sil²", "Pre²"],
    )
)
RHS = " + ".join(TERMS)
# A random-intercept variance on the zero boundary is optimiser noise whose
# exact value depends on library versions; below this fraction of the residual
# variance it is printed as ~0 so the output is stable.
BOUNDARY = 1e-6


def analyse(r: dict) -> dict:
    """Print the robustness report and return the values checked by make_figures."""
    values = {}
    df = r["df"].copy()
    df["config"] = df.groupby(["threshold", "silence_ms", "prefix_ms"]).ngroup()
    out = {}

    def col(res, label, terms=TERMS):
        out[label] = pd.DataFrame(
            {
                "coef": [res.params[t] for t in terms],
                "p": [res.pvalues[t] for t in terms],
            },
            index=[NAMES[t] for t in terms],
        )

    for y in ["fpr", "tl_ms"]:
        print(f"\n===== {y} =====")
        base = smf.ols(f"{y} ~ {RHS} + day", df).fit()
        col(base, "OLS (reported)")
        # (Cluster-robust SEs are not reported: with 19 clusters and regressors
        # that are constant within clusters they are known to be
        # anti-conservative.)
        # 2. mixed model, random intercept per design point
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mm = smf.mixedlm(f"{y} ~ {RHS} + day", df, groups=df["design_point"]).fit(
                reml=True
            )
        col(mm, "Mixed model (random intercept per design point)")
        vc = mm.cov_re.iloc[0, 0]
        icc = vc / (vc + mm.scale)
        vc_text = "~0 (boundary)" if vc < BOUNDARY * mm.scale else f"{vc:.3g}"
        print(
            f"mixed: random-intercept variance={vc_text}, "
            f"residual variance={mm.scale:.4g}, ICC={icc:.3f}"
        )
        values[f"{y}.icc"] = icc
        values[f"{y}.mixed_p"] = mm.pvalues
        # 3. fit on 19 design-point means (centre blocks kept separate, as the
        # reviewer asked)
        m19 = (
            df.groupby("design_point")
            .agg(
                {y: "mean", "x1": "first", "x2": "first", "x3": "first", "day": "mean"}
            )
            .reset_index()
        )
        fm = smf.ols(f"{y} ~ {RHS}", m19).fit()
        col(fm, "OLS on 19 design-point means")
        values[f"{y}.means19_p"] = fm.pvalues
        # 4. lack-of-fit against pure error (15 distinct configurations; model
        # without day)
        nd = smf.ols(f"{y} ~ {RHS}", df).fit()
        cm = df.groupby("config")[y].transform("mean")
        sspe = ((df[y] - cm) ** 2).sum()
        dfpe = len(df) - df["config"].nunique()
        sslof = nd.ssr - sspe
        dflof = int(nd.df_resid) - dfpe
        F = (sslof / dflof) / (sspe / dfpe)
        p = stats.f.sf(F, dflof, dfpe)
        print(
            f"lack of fit: SS_lof={sslof:.5g} (df {dflof}), SS_pe={sspe:.5g} "
            f"(df {dfpe}), F={F:.3f}, p={p:.3f}"
        )
        values[f"{y}.lof"] = (F, p, dflof, dfpe)
        # day vs replication collinearity
        tab = pd.crosstab(df["rep"], df["recording_date"])
        if y == "fpr":
            print("replication x recording date:\n", tab)
            values["rep_by_day"] = tab
        res = pd.concat(out, axis=1)
        print(res.round(4).to_string())
        out.clear()

    # 5. binomial model for FPR on trigger counts
    print("\n===== FPR, binomial GLM on counts =====")
    endog = np.column_stack(
        [df["vad_false_triggers"], df["vad_triggered"] - df["vad_false_triggers"]]
    )
    X = smf.ols(f"fpr ~ {RHS} + day", df).fit().model.exog
    names = smf.ols(f"fpr ~ {RHS} + day", df).fit().model.exog_names
    glm = sm.GLM(endog, X, family=sm.families.Binomial())
    b = glm.fit()
    disp = b.pearson_chi2 / b.df_resid
    # quasi-binomial: inflate SEs by sqrt(Pearson dispersion). (statsmodels'
    # scale="X2" mis-scales two-column count endog, so it is applied by hand.)
    p_quasi = 2 * norm.sf(np.abs(b.params / (b.bse * np.sqrt(disp))))
    print(
        f"binomial: Pearson dispersion = {disp:.3f} (1 = no overdispersion), "
        f"triggers per session: median {df.vad_triggered.median():.0f}, "
        f"range {df.vad_triggered.min()}-{df.vad_triggered.max()}"
    )
    t = pd.DataFrame(
        {
            "logit coef": b.params,
            "OR per SD": np.exp(b.params),
            "p binomial": b.pvalues,
            "p quasi-binomial": p_quasi,
        },
        index=names,
    )
    t = t.drop("Intercept").rename(
        index=lambda s: NAMES.get(s, s.replace("day", "Recording day"))
    )
    print(t.round(4).to_string())
    print("\nBonferroni threshold: p < 0.005")
    values["glm_dispersion"] = disp
    values["glm"] = t
    values["triggers"] = df["vad_triggered"]
    return values


def run(results: dict | None = None) -> dict:
    """Write robustness.txt and return the values checked by make_figures."""
    if results is None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            results = analyse_sweep.run(ROOT / "results", Path(tmp))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        values = analyse(results)
    OUTPUT.write_text(buf.getvalue())
    print(f"  Saved: {OUTPUT.name}")
    return values


if __name__ == "__main__":
    run()
