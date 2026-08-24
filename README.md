# ncaa-mbb-hoops-data

Python producer that reshapes [`ncaa-mbb-hoops-raw`](https://github.com/sportsdataverse/ncaa-mbb-hoops-raw)'s
parsed `stats.ncaa.org` men's basketball JSON into season-level tidy datasets.
The upstream source is `stats.ncaa.org` (via the bigballR port in sdv-py) --
**not** ESPN; NCAA contest ids are strings, not ESPN ints. Sister repo to the
wehoop (WNBA) and hoopR (NBA/WBB) data producers -- same build -> publish
shape, different sport/league.

## Datasets

Eleven datasets, keyed in `config.REGISTRY`. Six are DIRECT extracts of a
top-level key in each game's parsed JSON; the other five are DERIVED — built
from those same parsed payloads, from the raw roster files, or from the
crosswalk, rather than from one named family.

Listed in stage order. That order is `config.REGISTRY` insertion order, which
`--dataset all` iterates, so it is also the order a full build runs in:
identity/reference frames first, then per-game events and box, then the
lineup-grain frames. It is a **reading order, not a dependency chain** — no
dataset is built from another dataset's OUTPUT, so any one can be built alone
in any order (`--dataset shots` works on its own).

| NN | dataset | type | description |
| --- | --- | --- | --- |
| 01 | `team_ids` | derived | stats.ncaa.org team-id crosswalk for the season, from the bundled sdv-py `ncaa_mbb_team_ids` table. Reads no games at all. |
| 02 | `schedule` | derived | One row per game (home/away/date/final score). Built from each payload's `pbp` **family** — the parsed JSON has no schedule family. |
| 03 | `team_rosters` | derived | Per-team season rosters, read from the raw repo's captured roster JSON (not from a parsed-game family). |
| 04 | `rosters` | derived | Distinct `(team, player)` pairs per season with a games-played count. Built from each payload's `player_box` **family**, because the parsed tree has no roster family and sdv-py's roster parser needs roster HTML this tree doesn't hold. |
| 05 | `pbp` | direct | Play-by-play, one row per event. |
| 06 | `player_box` | direct | Per-player box score, one row per player/game. |
| 07 | `team_box` | direct | Per-team box score, one row per team/game. |
| 08 | `lineups` | direct | On-court five-man units by stint. |
| 09 | `matchup_stints` | derived | One row per constant-10-man floor segment, with score/possession deltas. `home_lineup_key`/`away_lineup_key` join to `lineups.lineup_key`. |
| 10 | `possessions` | direct | Possession-level rollup. |
| 11 | `shots` | direct | Shot events with location. |
| 99 | *(schedule master)* | cross-dataset | Not a dataset — the D34 coverage index over all of the above. See below. |

Two of the reference frames read a per-game **family** rather than a dedicated
one: `schedule` from `pbp` and `rosters` from `player_box`. That is a content
lineage, not a build dependency — both re-derive from the raw payloads, so
they are still buildable before stages 05/06 ever run. They sit early because
they are dimension tables you join everything else to.

## Schedule master (stage 99)

Three committed artifacts answer "what does this repo actually have?", all
emitted in one pass by `python/ncaa_mbb_99_schedule_master_creation.py`:

| file | grain | what it is |
| --- | --- | --- |
| `mbb/ncaa_mbb_schedule_master.parquet` | one row per contest | **Denominator** — every contest stats.ncaa.org lists, including ones nothing was built from. |
| `mbb/ncaa_mbb_games_in_data_repo.parquet` | one row per contest | **Numerator** — only contests present in ≥1 dataset. Join consumer work against this one. |
| `mbb/ncaa_mbb_schedule_coverage.parquet` | one row per season | Game count, date span, and `pct_in_*` per dataset. |

The denominator comes from the **raw** repo's `mbb/mbb_schedule_master.parquet`
(D33), not from the built `schedule` dataset — that one is derived from `pbp`,
so using it would make coverage 100% by construction. Each `in_*` flag is
stamped from the committed per-season parquet of that dataset, and the flag SET
is derived from `config.REGISTRY` (`level == "game"`), never hand-listed. Every
flag is materialized, so a family with no coverage reads as zeros rather than
disappearing from the schema.

```bash
NCAA_MBB_RAW_ROOT=../ncaa-mbb-hoops-raw \
  uv run python python/ncaa_mbb_99_schedule_master_creation.py
```

## Run order

1. **Build** -- reshapes the raw JSON and writes parquet in-repo under
   `mbb/{dataset}/parquet/ncaa_mbb_{dataset}_{season}.parquet` (committed).
2. **Publish** -- uploads parquet + csv.gz + rds as release assets to
   `sportsdataverse/sportsdataverse-data` (not committed; requires `gh` auth).
3. **Stage 99** -- after the seasons are built, rebuild the schedule master so
   the coverage index matches the tree.

```bash
# Build all 11 datasets for a season
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

### Historical backfill (whole-history rebuild)

`scripts/run_historical_publish.sh` builds **and publishes every season of every
dataset**. Ordinary incremental work is `run_build.sh` / `run_publish.sh` for a
single season -- reach for this only when re-materialising the full history.

```bash
bash scripts/run_historical_publish.sh                  # 2010..2026, all datasets
START=2015 END=2010 bash scripts/run_historical_publish.sh
DATASETS="pbp shots" bash scripts/run_historical_publish.sh
DRY_RUN=1 bash scripts/run_historical_publish.sh        # build + stage, no uploads
FORCE=1 bash scripts/run_historical_publish.sh          # rebuild even if current

tail -f logs/historical_publish_<timestamp>.log         # watch it live
```

It is **resumable**: a `(dataset, season)` is skipped only when its parquet
exists *and* the committed manifest agrees -- a 0-byte or orphaned parquet
rebuilds rather than being trusted. Ctrl-C is safe. One failing dataset-season
never aborts the sweep; the run exits RED at the end and lists what failed.

**Use `FORCE=1` when the build logic has changed**, since the resume check
proves a file is *present*, not that it was produced by the current code. The
first full publish needed it: the committed 2025-26 parquet predated the
identity-enrichment columns and had 55 columns where a fresh build has 87.

It auto-selects an R install that actually has `arrow` for the `.rds` assets
rather than trusting `Rscript` on `PATH` (on the maintainer's box PATH resolves
to R-4.5.3, which has no `arrow`, and the only symptom is a warning plus a
silently missing asset).

## Format policy

- **parquet**: committed in-repo under `mbb/{dataset}/parquet/` as
  `ncaa_mbb_{dataset}_{season}.parquet`, always written on every build.
  The `ncaa_mbb_` prefix matches the release tag, so a downloaded asset keeps
  its provenance in the filename instead of colliding with every other
  league's `pbp_2026.parquet`.
- **parquet + csv.gz + rds**: published as release assets to
  `sportsdataverse/sportsdataverse-data`, tagged `ncaa_mbb_{dataset}` (e.g.
  `ncaa_mbb_pbp`). Uploaded one file at a time via
  `gh release upload <tag> <file> --repo sportsdataverse/sportsdataverse-data --clobber`,
  creating the release if it doesn't exist yet. csv/rds are staged under the
  gitignored `mbb/_release_build/` and are re-derivable from the committed
  parquet -- they are never committed.
- **The release csv is GZIPPED (`.csv.gz`), deliberately.** One season of `pbp`
  is ~3.1M rows and writes a **2,126,337,961-byte** plain csv -- 99.0% of
  GitHub's 2 GiB (2,147,483,648-byte) per-asset hard limit, so a season slightly
  longer than 2025-26 would fail to upload outright. Gzipped it is
  **99,701,928 bytes (~95 MiB)**, 21.3x smaller. (Byte counts rather than
  rounded GB/MB: the limit being cleared is measured in bytes.)
  `espn_cfb_model_pbp` already ships `.csv.gz` on the same release repo. Read
  one with `pl.read_csv(gzip.open(path, "rb"))`, or `readr::read_csv()` in R,
  which decompresses transparently.
- **rds** requires R with the `arrow` package (`Rscript` shells out to
  `arrow::read_parquet` -> `saveRDS`). Resolution order: `SDV_RSCRIPT` env,
  then `RSCRIPT` env, then `Rscript` on `PATH`, then a scan of
  `C:/Program Files/R/R-*/bin/Rscript.exe`. RDS conversion failure (e.g. no R
  install has `arrow`) only logs a warning -- it never blocks the parquet+csv.gz
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
`tests/fixtures/raw_root/mbb/json/` plus an `mbb_schedule_master.parquet`
for season `2026`. `team_ids` reads the bundled sdv-py crosswalk, so it's
offline too.

```bash
uv run pytest -q
```

`tests/test_e2e.py` builds all 11 datasets from the fixtures into a temp
directory and asserts each parquet is written, non-empty, schema-stable
across the write/read round-trip, and holds the dtype-discipline contract
(`contest_id`/`id` as Utf8, `season` as Int64).

## sdv-py loader wiring (deferred)

Wiring these datasets into sdv-py's `mbb_loaders` is a follow-up task, done
*after* the first real publish. The loader introspects the live published
parquet's footer schema, so it can't be generated until a real `ncaa_mbb_*`
release exists on `sportsdataverse/sportsdataverse-data`.

## Player name changes (`mbb/name_changes/`)

stats.ncaa.org re-renders roster and box pages with a player's **current** name,
while the play-by-play preserves the name **as it was at game time**. A player
who changes their name therefore never matches between `possessions` and
`team_rosters`, in any season, and no safe string rule bridges
`KATELYNN.LIMARDO -> KATELYNN.MARTIN`.

The `box_score` page binds both renderings to one numeric player id:

```text
shot JS   addShot(..., '... player_768547579 team_201', ...)
          "made by Miah Monahan(Eastern Ill.)"      <- game-time
dropdown  <option value="768547579">Miah Meyer      <- current
```

`ops/build_name_changes.py` extracts that binding across the whole raw tree:

```sh
python ops/build_name_changes.py --league mbb
```

~3 minutes over 99,932 games -> 1,085 name-changes, written to
`mbb/name_changes/parquet/ncaa_mbb_name_changes.parquet`
(`season`, `team`, `name_game_time`, `name_current`, `n_games`).

**Known gap: 2019+ only.** The binding is the shot-chart JS, and shot charts
start in 2019 -- the same boundary that makes `shots` a 2019+ dataset. Earlier
seasons still benefit where a career spans the boundary (WBB 2018 +2.06pp,
2017 +0.72pp of fully-resolved possessions), but 2016 and older gain nothing.

Only rows whose two **coded** names differ are emitted; comparing raw HTML
strings yields false positives from entity/whitespace noise.

**Not yet a published dataset** -- it is a committed artifact consumed by the
sdv-py RAPM identity layer. Registering it in `config.REGISTRY` (and so on a
release tag) is a separate decision.
## RAPM (`ops/build_rapm.py`)

Feeds the hoop-explorer RAPM engine (`sportsdataverse.mbb.mbb_rapm`) from the
raw HTML. The engine consumes ES-derived lineup buckets -- 257 keys each --
which **no published dataset carries** (`possessions` is 56 flat columns,
`lineups` 77), so the buckets come from the chain that produces them:

```text
get_box_lineup -> create_lineup_data -> lineup_stats_buckets
  -> lineup_to_team_report -> build_player_context
  -> calc_player_weights / calc_lineup_outputs / slow_regression -> calculate_rapm
```

The call sequence is copied from sdv-py's committed end-to-end test, so the
bucket shape is right by construction rather than inferred.

```sh
uv run python ops/build_rapm.py --league mbb --season 2024 --workers 8
```

### Publishing it (`ops/publish_rapm.py`)

Stage 3. `build_rapm.py` emits a frame keyed on team plus a DISPLAY name; the
publisher attaches `season` / `team_id` / `player_id` / `person_id` and uploads
the `ncaa_mbb_rapm_within_team` release dataset via `sportsdataverse.release`.

```sh
# dry run (default) -- runs the full join and enforces the floor, uploads nothing
uv run python ops/publish_rapm.py --league mbb --rapm-dir ops/out

# publish
uv run python ops/publish_rapm.py --league mbb --rapm-dir ops/out --publish
```

**The estimand is WITHIN-TEAM**, not league-wide -- the tag name says so, and
`ncaa_mbb_rapm` stays free for a future league-wide (Path B) dataset.

Publishing is gated: a 99% id match-rate FLOOR that `--min-match-rate` may raise
but never lower, and a hard refusal when the name-change crosswalk is missing
(without it a renamed player silently becomes two `person_id`s and the match
rate cannot detect it). Ambiguity is nulled, never guessed.

Note `sportsdataverse_save` uploads but never CREATES a release -- the tag must
exist first (`gh release create`).

Each build also writes `ncaa_mbb_rapm_<season>.manifest.json` recording what the run actually
covered (partial flag, team/limit, games_processed vs games_available, teams
rated, rows). The publisher REFUSES a season whose manifest is missing, marks it
partial, shows a truncated run, or disagrees with the parquet's row count.

The filename suffix only proves a run was *declared* partial; it cannot prove a
run that claimed to be full actually finished. An interrupted full run writes
the canonical name with fewer teams and still clears the match-rate floor.

`--allow-unmanifested` covers the pre-manifest corpus only. It waives the proof
rather than supplying one, and never silences a manifest that says PARTIAL or
TRUNCATED.

~0.51 s/game single-threaded; 8 workers does a season in ~9 min.

**D-I scoping is on by default and is not cosmetic.** Rating every team that
appears on the floor wrecks the distribution, because non-D-I exhibition
opponents play one or two tracked games each. The figures below were **measured
on the WBB twin (2024)**, not on MBB -- the mechanism is shared, but the MBB
numbers have not been measured yet and are NOT assumed equal:

| scope | n | mean | sd | max abs |
| --- | --- | --- | --- | --- |
| all teams | 5,836 | -1.85 | 5.07 | 33.3 |
| **D-I only** | **4,081** | **-0.16** | **2.18** | **9.4** |
| non-D-I | 1,755 | -6.08 | 6.98 | 33.3 |

D-I alone centres at ~0 with sd 2.18 -- the shape RAPM should have -- and the
WBB leaderboard resolves to real elite players (Brink, Cardoso, Fulwiley, Ejim).
`--all-teams` disables the scope.

**This engine's RAPM is WITHIN-TEAM**: it apportions one team's performance
across its own players, a different estimand from league-wide RAPM. Provisional
-- not yet oracle-gated against Torvik/KenPom, and not published.
