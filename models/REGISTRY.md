# Model registry

One row per published RAPM estimand (Track C step 1). Gate definitions live in
the fitting scripts' docstrings (`ops/build_rapm_league.py`); floors were
frozen from the 2026-08-24 full validation sweep (34 league-seasons,
non_di=drop, lambda=1000) and are **never lowered** — `--min-spearman` may
only RAISE them. `tests/test_model_registry.py` keeps this table in lockstep.

| model | artifact(s) | release tag | training data | fitting script | gates at publish | last retrain | cadence |
|---|---|---|---|---|---|---|---|
| League-wide RAPM (Path B — every D-I player on one scale) | per-season parquet, **52 assets** | `ncaa_mbb_rapm` | this repo's published `possessions` + `team_rosters` + `name_changes` trees, 2011–2026 (2010 excluded by gate 1, not by a season list) | `ops/build_rapm_league.py` (stages 1–2) → `ops/publish_rapm_league.py` (stage 3) | publish-blocking, a failed season writes NOTHING: usable-possession ≥ **0.65** (obs 2011+ min 0.7414); intercept era band **[95, 112]** (obs 99.24–107.97) + hca **[1.0, 4.0]** — scale-bug catchers Spearman can't see; Torvik external: ≥ 250 joined teams AND Spearman(team_net, adjem) ≥ **0.93** (obs min 0.9434) | 2026-08-24 | manual (`ops/`) — **scheduled retrain NOT wired; Track C follow-up** |
| Within-team RAPM (Path A — apportions one team's performance across its players) | per-season parquet, **55 assets** | `ncaa_mbb_rapm_within_team` | raw NCAA HTML bundles via the sdv-py hoop-explorer engine (`mbb_rapm.build_player_context` + ridge), ES lineup buckets rebuilt from the parse chain | `ops/build_rapm.py` → `ops/publish_rapm.py` | shape proven by the committed e2e (`tests/mbb/test_mbb_ncaa_lineup_aggregation_e2e.py` in sdv-py); **different estimand** from league-wide — `estimand` column stamped into every row; no Torvik gate (not comparable) | 2026-08-24 | manual (`ops/`) — scheduled retrain NOT wired |

Notes:
- The two rows are deliberately different estimands and cross-check each
  other; never merge their tags.
- Oracle fixture: `ops/oracle/ncaa_mbb_torvik.parquet` (join on the
  stats.ncaa.org name). A NaN rho or an undersized/missing oracle season is a
  FAILURE, never a skip.
