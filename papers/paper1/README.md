# Paper 1

Manuscript accompanying this repository's Phase 1 methodology:

> *Endpointing Latency versus Interruptive False Positives in Server-Side VAD:
> A Response-Surface Characterisation of a Production Speech-to-Speech API*

Under review at the International Journal of Speech Technology. Citation
details will be added here once available.

## Regenerating the figures

From the repository root:

```bash
pip3 install -r sources/requirements.txt
python3 papers/paper1/make_figures.py
```

This runs [`sources/analyse_sweep.py`](../../sources/analyse_sweep.py) on the
76 study sessions in [`results/`](../../results/), then the Appendix A
robustness analyses in [`robustness.py`](robustness.py), and writes the
outputs to [`figures/`](figures/). It then checks every value the manuscript reports
against the regenerated analysis and exits with an error if any of them cannot
be reproduced at its reported precision. The bootstrap uses a fixed seed, so a
re-run reproduces the committed files byte for byte.

| File | Content |
|------|---------|
| `Fig1.eps/.tif/.png` | Response surfaces for FPR and TL |
| `Fig2.eps/.tif/.png` | FPR versus TL: sessions, configuration means and non-dominated sets |
| `Fig3.eps/.tif/.png` | Standardised RSM coefficients |
| `Fig4.eps/.tif/.png` | Marginal effects |
| `summary.txt` | Descriptives, coefficient tables with and without the day covariate, secondary outcomes, bootstrap non-dominance frequencies |
| `pareto_model.json` | Non-dominated configuration means, as loaded by `adaptive_tuner.py` |
| `robustness.txt` | Robustness checks for Appendix A: mixed model, design-point means, lack of fit, binomial FPR model |

## Figure captions

**Fig. 1** Response surfaces of the second-order model for (a) interruptive
false-positive rate and (b) endpointing latency as functions of activation
threshold and silence duration, with prefix padding fixed at 300 ms; open
circles mark design-point locations projected onto this plane

**Fig. 2** Interruptive false-positive rate versus endpointing latency for all
sessions and the 15 configuration means (replicated centre point pooled,
n = 20) with 95% bootstrap confidence intervals, with the non-dominated
configuration means ringed and joined by a step line and the non-dominated
single sessions circled

**Fig. 3** Standardised coefficients with 95% confidence intervals of the
second-order response surface models for (a) interruptive false-positive rate
and (b) endpointing latency, with filled bars marking terms significant at the
Bonferroni-corrected level (p < 0.005); the recording-day coefficient is the
difference between the second and first day

**Fig. 4** Marginal effects of activation threshold, silence duration and
prefix padding on (a–c) interruptive false-positive rate and (d–f) endpointing
latency, showing individual sessions and level means with 95% bootstrap
confidence intervals
