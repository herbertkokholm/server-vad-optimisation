# Contributing

Contributions are welcome, particularly cross-domain replications.

## Sharing domain results

If you run a sweep in a new domain, a pull request with the following is ideal:

- Raw JSONL output in `example_data/` (or a subdirectory), or a link to it if the data is large
- Output from `analyse_sweep.py` — at minimum `summary.txt` and `pareto_model.json`
- A brief note in the PR description: domain, speaker population, any deviations from the standard protocol

If you would rather not share the data publicly, open an issue instead — methodology questions and cross-domain observations are equally valuable.

## Bug reports and methodology questions

Open an issue. For bugs, include the Python version, a minimal reproducing command, and any relevant error output.

## Code changes

Small fixes (typos, broken imports, analysis edge cases) can go straight to a PR. For anything that changes the sweep design or RSM fitting, open an issue first so the implications for reproducibility can be discussed.

## Code style

This project has no linter configuration. Match the style of the file you are editing — the codebase is intentionally minimal.
