# Contributing

Thanks for helping. Pseudorca is small on purpose; please open an issue before large changes.

## Development

```bash
git clone https://github.com/pomcho555/pseudorca && cd pseudorca
uv sync --all-extras          # creates .venv with the dev tools
uv run pytest                 # tests
uv run ruff check src tests   # lint
uv run ruff format src tests  # format
```

## Ground rules

- Read `DESIGN.md` first. It is the contract between modules; update it in the same change when you
  change behaviour.
- **Never execute notebook content.** No `exec`, `eval`, imports or `pickle.loads` of anything taken from
  a notebook, a payload or an archive. This applies to tests too.
- **Tests use synthetic data only** and make no network calls (fake the LLM and Kaggle clients).
- Keep `pseudorca --help` fast: import heavy libraries (numpy, openai, anthropic, kaggle) inside the
  functions that need them.
- Kaggle access stays read-only.
