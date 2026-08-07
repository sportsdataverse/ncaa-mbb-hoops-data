# ncaa-mbb-hoops-data Copilot Instructions

## Project Context

Python producer for the `stats.ncaa.org` NCAA men's basketball release
datasets. It reshapes the parsed per-game JSON from `ncaa-mbb-hoops-raw`
into season-level tidy parquet/csv and publishes them as release assets on
`sportsdataverse/sportsdataverse-data`.

Pipeline: `stats.ncaa.org -> ncaa-mbb-hoops-raw -> ncaa-mbb-hoops-data [HERE] -> sportsdataverse-data`.

**This repo reshapes; it does not scrape.** It is fully offline — it reads a
sibling `ncaa-mbb-hoops-raw` checkout (or `NCAA_MBB_RAW_ROOT`) and never hits
`stats.ncaa.org`. Wanting to fetch a page here means the work belongs in
`-raw`. Never mix the two stages.

## Repository Workflow

- Branch from `main`; `main` is the default branch.
- The build package is `python/ncaa_mbb_data_build/` (cli, config, build,
  ingest, derived, reshapers, io, publish, rds). `config.REGISTRY` is the
  dataset registry — some datasets are direct extracts of a parsed-JSON key,
  others are derived from other datasets. See `README.md` for the table.
- `python/ncaa_mbb_NN_*_creation.py` are numbered stage shims (01..11).
  The numbers are **dataset identity, not execution order**;
  `tests/test_stage_inventory.py` gates the set.

## ⚠️ The schedule-master name fallback

`ingest.py` tries `("mbb_schedule_master.parquet", "schedule_master.parquet")`
in that order. The prefixed name (D33/D36) is canonical; the legacy unprefixed
one is a fallback that exists because the **writer** (`ncaa-mbb-hoops-raw` and
sdv-py's `scrape/ncaa/discover.py`) still emits the old name. **When the writer
renames, the fallback drops** — don't remove it before then.

## Build & Development Commands

```sh
uv sync --frozen
uv run pytest -q
uv run ruff check python tests

SEASON=2026 bash scripts/run_build.sh                 # build (offline)
SEASON=2026 DATASET=shots bash scripts/run_build.sh
SEASON=2026 bash scripts/run_publish.sh               # build + upload via gh
```

`run_build.sh` defaults `NCAA_MBB_RAW_ROOT` to `../ncaa-mbb-hoops-raw`; an
already-set value wins. Both drivers tee to `logs/run_*_<timestamp>.log`.

## Code Style

- Follow the parent SDK's Python conventions: `snake_case`, 4-space indent.
- Deps live in `pyproject.toml` + `uv.lock` (no `requirements.txt`);
  `sportsdataverse` is pinned to git `main` via `[tool.uv.sources]` and CI
  installs with `uv sync --frozen`. It is a **git source, not a `../../sdv-py`
  path** — a relative pin only builds on a machine with this sibling layout and
  fails on CI. For local sdv-py work: `uv pip install -e ../../sdv-py`.
- polars 1.x. ruff is pinned: `select = ["E4","E7","E9","F","I"]`,
  `ignore = ["E712"]` (polars bool masks are written `pl.col("c") == True` on
  purpose); isort `known-first-party = ["ncaa_mbb_data_build"]`.
- Tests live in `tests/` at repo **root**, fixtures in `tests/fixtures/`;
  pytest is wired with `testpaths = ["tests"]`.
- Every script in `scripts/` must be referenced by a runbook, workflow, or
  another script — the shared `orphan-scripts` gate fails otherwise.

## CI

- `tests.yml` — sparse-checkout (`python`, `tests`, `pyproject.toml`,
  `uv.lock`), `uv sync --frozen`, `ruff check python tests`, `pytest -q`.
- `orphan_scripts.yml` — the shared `sportsdataverse/.github` orphan-scripts gate.

## Cross-Repo References

- Upstream scraper: <https://github.com/sportsdataverse/ncaa-mbb-hoops-raw>
- WBB twin: <https://github.com/sportsdataverse/ncaa-wbb-hoops-data>
- SDK internals: <https://github.com/sportsdataverse/sportsdataverse-py/blob/main/CLAUDE.md>

## Conventional Commits

Use: `type(scope): description`. Common types: `feat`, `fix`, `chore`, `ci`, `docs`, `refactor`, `test`. Use `type!:` or a `BREAKING CHANGE:` footer for breaking changes.

**Important: Never include AI agents or assistants (e.g., Claude, Copilot, Cursor, GPT, Gemini) as co-authors on commits.** Omit all `Co-Authored-By` trailers referencing AI tools. This applies whether the change was generated, refactored, or reviewed with AI assistance — the human author is the sole attributable contributor.
