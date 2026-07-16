# ncaa-mbb-hoops-data

Python producer that reshapes [`ncaa-mbb-hoops-raw`](https://github.com/sportsdataverse/ncaa-mbb-hoops-raw)'s
parsed `stats.ncaa.org` men's basketball JSON into season-level tidy datasets.
The upstream source is `stats.ncaa.org` (via the bigballR port in sdv-py) --
**not** ESPN; NCAA contest ids are strings, not ESPN ints. Sister repo to the
wehoop (WNBA) and hoopR (NBA/WBB) data producers -- same build -> publish
shape, different sport/league.

## Datasets

Nine datasets, keyed in `config.REGISTRY`. Six are DIRECT extracts of a
top-level key in each game's parsed JSON; three are DERIVED from other
datasets rather than a parsed-JSON family.

| dataset | type | description |
|---|---|---|
| `pbp` | direct | Play-by-play, one row per event. |
| `lineups` | direct | On-court five-man units by stint. |
| `possessions` | direct | Possession-level rollup. |
| `player_box` | direct | Per-player box score, one row per player/game. |
| `team_box` | direct | Per-team box score, one row per team/game. |
| `shots` | direct | Shot events with location. |
| `schedule` | derived | One row per game (home/away/date/final score), built from that game's `pbp`. |
| `rosters` | derived | Distinct `(team, player)` pairs per season, with a games-played count. Built from `player_box` because the parsed-JSON tree has no dedicated roster family -- sdv-py's roster parser needs separately-captured roster HTML this tree doesn't hold. |
| `team_ids` | derived | stats.ncaa.org team-id crosswalk for the season, from the bundled sdv-py `ncaa_mbb_team_ids` table. |

## Run order

1. **Build** -- reshapes the raw JSON and writes parquet in-repo under
   `mbb/{dataset}/parquet/{dataset}_{season}.parquet` (committed).
2. **Publish** -- uploads parquet + csv + rds as release assets to
   `sportsdataverse/sportsdataverse-data` (not committed; requires `gh` auth).

```bash
# Build all 9 datasets for a season
uv run python -m ncaa_mbb_data_build build --dataset all --season 2026

# Build one dataset
uv run python -m ncaa_mbb_data_build build --dataset shots --season 2026

# Build + publish (uploads release assets)
uv run python -m ncaa_mbb_data_build build --dataset all --season 2026 --publish
```

Or the launcher scripts, which set up logging and the raw-root env:

```bash
SEASON=2026 bash scripts/run_build.sh
SEASON=2026 DATASET=shots bash scripts/run_build.sh   # single dataset

SEASON=2026 bash scripts/run_publish.sh                # build + publish
```

`NCAA_MBB_RAW_ROOT` points at the sibling `ncaa-mbb-hoops-raw` checkout (the
launchers default it to `../ncaa-mbb-hoops-raw`); an HTTP fallback is used
when that checkout isn't available locally.

## Format policy

- **parquet**: committed in-repo under `mbb/{dataset}/parquet/`, always
  written on every build.
- **parquet + csv + rds**: published as release assets to
  `sportsdataverse/sportsdataverse-data`, tagged `ncaa_mbb_{dataset}` (e.g.
  `ncaa_mbb_pbp`). Uploaded one file at a time via
  `gh release upload <tag> <file> --repo sportsdataverse/sportsdataverse-data --clobber`,
  creating the release if it doesn't exist yet. csv/rds are staged under the
  gitignored `mbb/_release_build/` and are re-derivable from the committed
  parquet -- they are never committed.
- **rds** requires R with the `arrow` package (`Rscript` shells out to
  `arrow::read_parquet` -> `saveRDS`). Resolution order: `SDV_RSCRIPT` env,
  then `RSCRIPT` env, then `Rscript` on `PATH`, then a scan of
  `C:/Program Files/R/R-*/bin/Rscript.exe`. RDS conversion failure (e.g. no R
  install has `arrow`) only logs a warning -- it never blocks the parquet+csv
  upload.

## Requirements / credentials

- [`uv`](https://docs.astral.sh/uv/) for everything -- never bare `python`/`pip`.
- The `sportsdataverse` dependency resolves to the local `../../sdv-py`
  sibling checkout (editable): NCAA parsers aren't on PyPI yet. Swap to the
  PyPI pin in `pyproject.toml` once a release ships NCAA support.
- Publishing needs `gh` authenticated with a token: `GH_TOKEN`, `GITHUB_PAT`,
  or `SDV_GH_TOKEN` (checked in that order; `run_publish.sh` also falls back
  to `~/.Renviron`).
- RDS conversion needs an R install with `arrow` on `PATH` (or `SDV_RSCRIPT`
  pointed at one).

## Tests

Hermetic, offline, no network: 8 fixture games under
`python/tests/fixtures/raw_root/mbb/json/` plus a `schedule_master.parquet`
for season `2026`. `team_ids` reads the bundled sdv-py crosswalk, so it's
offline too.

```bash
uv run pytest python/ -q
```

`python/test_e2e.py` builds all 9 datasets from the fixtures into a temp
directory and asserts each parquet is written, non-empty, schema-stable
across the write/read round-trip, and holds the dtype-discipline contract
(`contest_id`/`id` as Utf8, `season` as Int64).

## sdv-py loader wiring (deferred)

Wiring these datasets into sdv-py's `mbb_loaders` is a follow-up task, done
*after* the first real publish. The loader introspects the live published
parquet's footer schema, so it can't be generated until a real `ncaa_mbb_*`
release exists on `sportsdataverse/sportsdataverse-data`.
