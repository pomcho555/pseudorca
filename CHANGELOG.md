# Changelog

## Unreleased

- `effects`: which ideas each submission has (from the verdicts on its notebook), and what each idea adds
  to the score. An additive least-squares model over the scored submissions (residual-SD t intervals, or a
  known `--noise`) and over the local runs of each protocol (paired two-stage bootstrap). Ideas that always
  change together are measured together; what the submissions cannot separate is reported, never guessed.
  `--sub` breaks a submission's score down idea by idea.
  Local runs add the run-to-run noise that runs with the same ideas show. A lack-of-fit test flags
  submissions that stray from the model more than resubmissions do. Defaults for the score's direction
  and a known noise can live in `[effects]` in `config.toml`.
- `report` draws the same: a submissions × ideas matrix, a leaderboard chart, forest plots and a breakdown
  (new options `--metric` and `--resamples`).
- `audit --submitted` audits every notebook that has a submission, skipping (with a warning) those it
  cannot read, and counts LLM failures.
- Terminal output replaces control characters and bidirectional overrides in ledger strings with `?`.
- A JSONL results table nested too deeply is a one-line error instead of a crash.
- First version: event-sourced ledger in git, static notebook extraction, fingerprint, patch and LLM
  audits with verified quotes, `check` before submitting, read-only Kaggle sync and pull, paired bootstrap
  comparison of local runs, leaderboard noise, and a single-file HTML report.
