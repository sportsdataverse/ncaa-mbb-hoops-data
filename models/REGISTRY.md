# Model registry

One row per published RAPM estimand (Track C step 1). Gate definitions live in
the fitting scripts' docstrings (`ops/build_rapm_league.py`); floors were
frozen from the 2026-08-24 full validation sweep (34 league-seasons,
non_di=drop, lambda=1000) and are **never lowered** — `--min-spearman` may
only RAISE them. `tests/test_model_registry.py` keeps this table in lockstep.

| model | artifact(s) | release tag | training data | fitting script | gates at publish | last retrain | cadence |
|---|---|---|---|---|---|---|---|
| League-wide RAPM (Path B — `estimand` = every D-I player on one scale) | per-season parquet, **52 assets**; additive columns `orapm_se` / `drapm_se` / `rapm_net_se` (ridge-posterior SEs, 2026-09-01) | `ncaa_mbb_rapm` | this repo's published `possessions` + `team_rosters` + `name_changes` trees, 2011–2026 (2010 excluded by gate 1, not by a season list) | `ops/build_rapm_league.py` (stages 1–2) → `ops/publish_rapm_league.py` (stage 3) | publish-blocking, a failed season writes NOTHING: usable-possession ≥ **0.65** (obs 2011+ min 0.7414); intercept era band **[95, 112]** (obs 99.24–107.97) + hca **[1.0, 4.0]** — scale-bug catchers Spearman can't see; Torvik external: ≥ 250 joined teams AND Spearman(team_net, adjem) ≥ **0.93** (obs min 0.9434); **SE gate (2026-09-01)**: σ̂² era band **[11000, 15000]** (obs 12,562–13,332), Spearman(poss, rapm_net_se) ≤ **−0.80** (obs −0.965 to −0.898), top-decile median SE < bottom-decile, split-half (odd/even games) coverage ≥ **0.95** under the posterior SE (obs ≥ 0.9995) and in **[0.92, 0.98]** under the sampling SE for O/D/net (obs 0.9465–0.9601, nominal 0.954) | 2026-09-01 (SE columns; coefficients unchanged) | annual cron + dispatch (`.github/workflows/ncaa_mbb_models.yml`) via `python/ncaa_mbb_model_01_rapm_league.py` |
| Within-team RAPM (Path A — apportions one team's performance across its players) | per-season parquet, **55 assets** | `ncaa_mbb_rapm_within_team` | raw NCAA HTML bundles via the sdv-py hoop-explorer engine (`mbb_rapm.build_player_context` + ridge), ES lineup buckets rebuilt from the parse chain | `ops/build_rapm.py` → `ops/publish_rapm.py` | shape proven by the committed e2e (`tests/mbb/test_mbb_ncaa_lineup_aggregation_e2e.py` in sdv-py); **different estimand** from league-wide — `estimand` column stamped into every row; no Torvik gate (not comparable) | 2026-08-24 | manual by design via `python/ncaa_mbb_model_02_rapm_within_team.py` — needs the raw NCAA HTML bundle checkout, not runner-friendly |

Notes:
- The published `*_se` are POSTERIOR standard errors. Writing `M = (X'WX+λI)⁻¹`, the per-component
  SEs are `orapm_se[i] = sqrt(σ̂²·M[i,i])` and `drapm_se[i] = sqrt(σ̂²·M[P+i,P+i])`. **`rapm_net_se`
  is NOT `sqrt(orapm_se² + drapm_se²)`** — `rapm_net = orapm + drapm` (drapm is already signed
  so higher = better defense), so its variance carries the O/D covariance term:

      rapm_net_se[i] = sqrt(σ̂²·(M[i,i] + M[P+i,P+i] + 2·M[i,P+i]))

  Reproducing the net interval from the two marginal SEs alone gives the WRONG width (O and D
  for the same player are estimated from the same possessions and are correlated). These are a
  credible interval for the true impact under the ridge prior (a low-minute player sits at
  ~0 ± σ̂/√λ). The frequentist sandwich COVARIANCE `σ̂²(M − λM²)` — whose SEs are
  `sqrt(diag(·))`, with the same +2·cov(O,D) term for net — is computed by the engine only to
  calibrate them (split-half gate 5d); it is not published because it collapses to ~0 for a
  player the ridge pins at zero.
- **They are conservative by ≈2.5×** relative to how far the estimate actually moves between two
  halves of a season (split-half z-sd ≈ 0.38, not 1.0) — λ=1000 is prior-dominated. `±2·SE` is a
  cautious band for the true impact; it is NOT the repeatability of the number, so gate 5d's ~1.0
  posterior coverage is a one-sided guard (SEs that shrank), never a claim of nominal calibration.
- The two rows are deliberately different estimands and cross-check each
  other; never merge their tags.
- Oracle fixture: `ops/oracle/ncaa_mbb_torvik.parquet` (join on the
  stats.ncaa.org name). A NaN rho or an undersized/missing oracle season is a
  FAILURE, never a skip.

## Evaluated, NOT adopted: RAPM stabilization levers (2026-09-02, PR #21)

Two levers were measured against the shipped flat-ridge baseline over ten held-out seasons
(2016–2025), hyperparameters frozen on 2014–2015, ~25% of games held out per season by
`contest_id % 4 == 0`. Harness: `ops/experiments/rapm_stabilization.py` (re-runnable);
decision rule applied in code by `ops/experiments/summarize_rapm_stabilization.py`; full
design and tables in the ClaudeCowork ledger
`2026-09-01-writeup-improvements/reports/rapm-stabilization.md`. Engine support (default-off)
is `sportsdataverse-py#436`.

| lever | out-of-sample game-margin MAE gain (13,081 held-out games) | seasons better | status |
|---|---|---|---|
| multi-year (stacked, `decay = 0.5`) | **0.095** pts/game, pooled game-cluster bootstrap 95% CI excludes 0 | 10/10 | measured, NOT the producer default |
| SPM prior (`prior_mean=`, exposure shrink `k = 100`) | **0.061** pts/game, CI excludes 0 | 10/10 | measured, NOT the producer default |
| both together | **0.135** pts/game, CI excludes 0 | 10/10 | measured, NOT the producer default |

**Why not adopted.** Both beat the baseline on the pre-registered criteria and hold the Torvik
floor, but flipping the producer's estimator republishes 52 live assets and first needs:

1. **A re-derived `σ̂²` gate band.** Gate 5(a) bounds `σ̂²` to `[11000, 15000]`; a decayed-weight pooled
   fit measures **7,733** against the single-season **13,212** (measured on MBB 2022-24, decay 0.5).
   That is not a bug — under decay weights `σ̂²` is a weighted residual variance on a different
   weight scale, i.e. a DIFFERENT quantity — but the band must be re-derived from its own full
   sweep. It must **not** be widened to let the change through.
2. **A season-`t` filter on the published frame.** A pooled fit rates 8,633 persons across three
   seasons where the single-season fit rates 4,919; without the filter, and without reporting
   season-`t` exposure, `off_poss` / `def_poss` silently become three-season sums.

**Finding worth keeping:** the multi-year gain is **not** concentrated below ~200 possessions,
which is what the original backlog item assumed. The absolute next-season-Spearman gain is flat
across playing time, so there is no possession threshold above which it stops helping — it is a
uniform variance reduction, not a tail stabiliser. The SPM prior *does* have a threshold, at
roughly **100 possessions**: it is worth +0.03…+0.06 Spearman above it and nothing below, because
under ~100 possessions a player's box *rates* are themselves noise.

**Not a blocker (measured, contrary to the 2026-09-01 design note):** the SE path survives pooling
— the 3-season design is dim 17,267 and `compute_se=True` completes in 24.4 s (vs 6.4 s
single-season). The engine's documented ~20k ceiling binds at a 4–5 season window.

## Operability (Track C steps 2–6)

- `models/manifest.yaml` — single home for the model/stage list (guarded by `tests/test_model_manifest.py`).
- One estimand = one numbered pipeline, flat in `python/` beside the data stages: `ncaa_mbb_model_01_rapm_league.py` / `ncaa_mbb_model_02_rapm_within_team.py`; run subsets with `scripts/ncaa_mbb_models.sh`.
- League-wide retrain is now WIRED: `.github/workflows/ncaa_mbb_models.yml` (dispatch + annual post-season cron). Within-team stays manual by design (raw HTML bundle dependency).
- Fingerprint skip: deliberately NOT used — inputs are living published trees; every run recomputes. Each run appends `models/ledger.jsonl`.
- Step 6: per-season parquet assets live on the release tags; nothing fitted is committed here.
