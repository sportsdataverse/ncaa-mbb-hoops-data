# NCAA MBB RAPM — model documentation

Two deliberately different estimands, published on separate tags (never merge):

| estimand | tag | assets | stage |
|---|---|---|---|
| League-wide (Path B — every D-I player on one scale) | `ncaa_mbb_rapm` | 52 per-season parquet | `python/ncaa_mbb_model_01_rapm_league.py` |
| Within-team (Path A — apportions one team's performance) | `ncaa_mbb_rapm_within_team` | 55 per-season parquet | `python/ncaa_mbb_model_02_rapm_within_team.py` |

## Features / design

Ridge regression (`lambda = 1000`) over the possession-level on/off design
matrix built from this repo's published `possessions` + `team_rosters` +
`name_changes` trees (league-wide) or the raw-bundle lineup reconstruction via
the sdv-py hoop-explorer engine (within-team). Seasons 2011-2026; 2010 is
excluded by the usable-possession gate, not by a season list.

## Gates + observed results (frozen from the 2026-08-24 full validation sweep)

Publish-blocking — a failed season writes NOTHING; floors may only be RAISED
(`--min-spearman` refuses lower values):

| gate | floor | observed |
|---|---|---|
| usable-possession fraction | >= 0.65 | 2011+ min 0.7414 |
| intercept era band (scale-bug catcher) | [95, 112] | 99.24-107.97 |
| home-court advantage band | [1.0, 4.0] | in-band all seasons |
| Torvik external (league-wide only) | >= 250 joined teams AND Spearman(team_net, adjem) >= 0.93 | min 0.9434 |

Within-team has NO Torvik gate (different estimand, not comparable); its shape
is proven by the committed sdv-py e2e lineup-aggregation test. Oracle fixture:
`ops/oracle/ncaa_mbb_torvik.parquet` — a NaN rho or missing oracle season is a
FAILURE, never a skip.

## Operability

League-wide retrain: `.github/workflows/ncaa_mbb_models.yml` (dispatch +
annual post-season cron). Within-team stays manual by design (needs the raw
HTML bundle checkout). Each run appends `models/ledger.jsonl`. Single home for
the stage list: `models/manifest.yaml`.

## Per-season results (real local sweep, 2026-09-01)

| season | usable % | players | intercept | HCA | Torvik rho (n) |
|---|---|---|---|---|---|
| 2011 | 77.38 | 4482 | 100.24 | 2.805 | 0.9598 (332) |
| 2012 | 79.90 | 4537 | 99.98 | 3.001 | 0.9706 (333) |
| 2013 | 75.38 | 4542 | 99.24 | 2.934 | 0.9591 (334) |
| 2014 | 77.14 | 4671 | 103.00 | 2.639 | 0.9584 (339) |
| 2015 | 78.09 | 4649 | 100.85 | 2.822 | 0.9617 (339) |
| 2016 | 75.07 | 4640 | 102.54 | 2.818 | 0.9587 (339) |
| 2017 | 75.58 | 4630 | 102.45 | 2.505 | 0.9653 (338) |
| 2018 | 74.14 | 4592 | 103.17 | 2.673 | 0.9509 (338) |
| 2019 | 78.87 | 4688 | 102.30 | 2.539 | 0.9719 (338) |
| 2020 | 83.05 | 4626 | 99.98 | 2.787 | 0.9711 (337) |
| 2021 | 84.36 | 4896 | 100.88 | 2.019 | 0.9434 (335) |
| 2022 | 86.70 | 4942 | 101.54 | 2.194 | 0.9770 (346) |
| 2023 | 85.83 | 4942 | 102.61 | 2.736 | 0.9714 (350) |
| 2024 | 84.87 | 4919 | 104.84 | 2.455 | 0.9725 (350) |
| 2025 | 83.30 | 4960 | 105.67 | 2.691 | 0.9652 (353) |
| 2026 | 89.84 | 4974 | 107.97 | 2.435 | 0.9783 (356) |

Card: [`ncaa_mbb_rapm_card.json`](ncaa_mbb_rapm_card.json)

## Figures

![RAPM distribution](figures/rapm_league_distribution_2026.png)

![Gate metrics by season](figures/rapm_league_gates_by_season.png)

![Team net vs Torvik](figures/rapm_vs_torvik_2026.png)
