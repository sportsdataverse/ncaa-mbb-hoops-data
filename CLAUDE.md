# CLAUDE.md — ncaa-mbb-hoops-data Development Guide

## Package Overview

This repo is the **reshape stage** for `stats.ncaa.org` men's college
basketball: it turns the parsed per-game JSON produced by
`ncaa-mbb-hoops-raw` into season-level tidy datasets (parquet/csv) and
publishes them as release assets on `sportsdataverse/sportsdataverse-data`.

Pipeline: `stats.ncaa.org -> ncaa-mbb-hoops-raw -> ncaa-mbb-hoops-data [HERE]
-> sportsdataverse-data`.

**The `-raw` / `-data` split is load-bearing: never mix them.** This repo is
**fully offline** — it reads a sibling `ncaa-mbb-hoops-raw` checkout (or
`NCAA_MBB_RAW_ROOT`) and never scrapes `stats.ncaa.org`. If you find yourself
wanting to fetch a page here, the work belongs in `-raw` instead. Conversely, a
bug in a *tidy dataset's shape* belongs here, not in `-raw`.

`README.md` carries the dataset table, the format policy, and the run order.

## Layout

```
python/
  ncaa_mbb_data_build/          # the build package (installed by uv sync)
    cli.py  config.py  build.py  ingest.py  derived.py
    reshapers.py  io.py  publish.py  rds.py  _logging.py  __main__.py
  ncaa_mbb_NN_*_creation.py     # numbered stage shims, 01..11
scripts/      # run_build.sh, run_publish.sh
tests/        # suite + fixtures/ at repo ROOT
logs/         # build/publish run logs
mbb/          # the built dataset tree (lineups, pbp, shots, team_box, …)
```

Stage order (= `config.REGISTRY` insertion order, which `--dataset all`
iterates, so it is also the order a full build runs in):

| NN | dataset | kind |
| --- | --- | --- |
| 01 | team_ids | derived (crosswalk only — reads no games) |
| 02 | schedule | derived |
| 03 | team_rosters | derived (raw roster files) |
| 04 | rosters | derived |
| 05 | pbp | direct |
| 06 | player_box | direct |
| 07 | team_box | direct |
| 08 | lineups | direct |
| 09 | matchup_stints | derived |
| 10 | possessions | direct |
| 11 | shots | direct |

That sequence is a **reading order** — identity/reference first, then per-game
events and box, then the lineup-grain frames — **not a dependency chain.** No
dataset is built from another dataset's output: every one is a pure function of
`(raw tree, season)`, so `--dataset shots` alone works. `matchup_stints` looks
like the exception and isn't: its `*_lineup_key` columns join to `lineups` when
you QUERY, but both are derived independently from the raw payloads when you
BUILD. `tests/test_stage_inventory.py` gates the shim set AND that the numbers
ascend with registry order — renumbering one without the other fails it.

`config.REGISTRY` is the dataset registry: 6 datasets are **direct** extracts
of a top-level key in each game's parsed JSON, the other 5 are **derived** from
the parsed payloads, the raw roster files, or the crosswalk. `README.md` has
the authoritative per-dataset table.

## The schedule-master name fallback (do not "clean up")

`ingest.py` resolves the season contest-id index by trying, in order:

```python
names = ("mbb_schedule_master.parquet", "schedule_master.parquet")
```

The prefixed `mbb/mbb_schedule_master.parquet` (D33/D36 master naming) is
**canonical**; the legacy unprefixed `mbb/schedule_master.parquet` is the
fallback. The fallback exists because the **writer** — `ncaa-mbb-hoops-raw`
and sdv-py's `scrape/ncaa/discover.py` — still emits the old name. **When the
writer renames, the fallback drops.** Removing it before then breaks ingest
against every existing raw checkout.

## Packaging

Root `pyproject.toml` + `uv.lock`. **There is no `requirements.txt`.**

- `sportsdataverse` is pinned to git `main` via `[tool.uv.sources]` — sdv-py's
  NCAA parsers (`ncaa_mbb_team_ids`, etc.) are not on PyPI yet. It is a **git
  source, not a `../../sdv-py` path**: a relative path pin makes the repo
  buildable only on a machine with this exact sibling layout, and fails on CI.
  For local sdv-py work, override with `uv pip install -e ../../sdv-py`.
- CI installs with `uv sync --frozen` — the lockfile is the contract, so a
  green run proves the build package works against the sdv-py revision this
  repo actually claims, not whatever is checked out next door.
- The build package installs from `python/` (`[tool.setuptools.packages.find]
  where = ["python"], include = ["ncaa_mbb_data_build*"]`) and exposes the
  `ncaa-mbb-data-build` console script.
- pytest: `testpaths = ["tests"]`.
- ruff: `select = ["E4","E7","E9","F","I"]`, `ignore = ["E712"]` (polars bool
  masks are written `pl.col("c") == True` on purpose), isort
  `known-first-party = ["ncaa_mbb_data_build"]`.

```sh
uv sync --frozen
uv run pytest -q
uv run ruff check python tests

SEASON=2026 bash scripts/run_build.sh                 # build (offline)
SEASON=2026 DATASET=shots bash scripts/run_build.sh
SEASON=2026 bash scripts/run_publish.sh               # build + upload via gh
```

`run_build.sh` defaults `NCAA_MBB_RAW_ROOT` to the sibling
`../ncaa-mbb-hoops-raw` checkout; an already-set value always wins. Both
drivers tee to `logs/run_{build,publish}_<timestamp>.log` — watch a running
job with `tail -f`.

## CI

- `.github/workflows/tests.yml` — sparse-checkout (`python`, `tests`,
  `pyproject.toml`, `uv.lock`; the built `mbb/` tree is never read by the
  tests, whose fixtures live in `tests/fixtures/`), then `uv sync --frozen`
  -> `ruff check python tests` -> `pytest -q`.
- `.github/workflows/orphan_scripts.yml` — the shared `sportsdataverse/.github`
  gate: every entry in `scripts/` must be referenced by a runbook, a workflow,
  or another script. Both `run_build.sh` and `run_publish.sh` are documented in
  `README.md` and above.

## Commit Convention

[Conventional Commits](https://www.conventionalcommits.org/):
`type(scope): description`. Common types: `feat`, `fix`, `chore`, `ci`, `docs`,
`refactor`, `test`, `build`. Use `type!:` or a `BREAKING CHANGE:` footer for
breaking changes.

**Never include AI agents or assistants (Claude, Copilot, Cursor, GPT, Gemini,
…) as co-authors.** Omit all `Co-Authored-By` trailers referencing AI tools,
whether the change was generated, refactored, or reviewed with AI assistance —
the human author is the sole attributable contributor. This is hook-enforced.

## Cross-Repo References

- Upstream scraper: `hoopR-dev/ncaa-mbb-hoops-raw`
- WBB twin: `wehoop-dev/ncaa-wbb-hoops-data`
- SDK internals: <https://github.com/sportsdataverse/sportsdataverse-py/blob/main/CLAUDE.md>
