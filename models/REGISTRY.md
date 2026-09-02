# Model registry

One row per published RAPM estimand (Track C step 1). Gate definitions live in
the fitting scripts' docstrings (`ops/build_rapm_league.py`); floors were
frozen from the 2026-08-24 full validation sweep (34 league-seasons,
non_di=drop, lambda=1000) and are **never lowered** — `--min-spearman` may
only RAISE them. `tests/test_model_registry.py` keeps this table in lockstep.

| model | artifact(s) | release tag | training data | fitting script | gates at publish | last retrain | cadence |
|---|---|---|---|---|---|---|---|
| League-wide RAPM (Path B — `estimand` = every D-I player on one scale) | per-season parquet, **52 assets**; additive columns `orapm_se` / `drapm_se` / `rapm_net_se` (ridge-posterior SEs, 2026-09-01); **estimator (2026-09-02)** = the **decayed-weight 3-season pool** (`decay = 0.5`, seasons *t−2…t*, columns keyed by the cross-season `person_id`, sliced back to season-*t* players and season-*t* exposure) for 2013+; 2011–2012 keep the flat single-season ridge (no full window) | `ncaa_mbb_rapm` | this repo's published `possessions` + `team_rosters` + `name_changes` trees, 2011–2026 (2010 excluded by gate 1, not by a season list) | `ops/build_rapm_league.py` (stages 1–2) → `ops/publish_rapm_league.py` (stage 3) | publish-blocking, a failed season writes NOTHING: usable-possession ≥ **0.65** (obs 2011+ min 0.7414); intercept era band **[95, 112]** (obs 99.24–107.97) + hca **[1.0, 4.0]** — scale-bug catchers Spearman can't see; Torvik external: ≥ 250 joined teams AND Spearman(team_net, adjem) ≥ **0.93** (obs min 0.9434); **SE gate (2026-09-01)**: σ̂² era band **[11000, 15000]** (obs 12,562–13,332), and for the pooled estimator its own **[11000, 15000]** re-derived from the 14-season 2026-09-02 pooled sweep (obs 12,646.5 in 2014 – 13,361.0 in 2026) by the same ±12.5%-and-round-outward rule that reproduces the flat band exactly, Spearman(poss, rapm_net_se) ≤ **−0.80** (obs −0.965 to −0.898), top-decile median SE < bottom-decile, split-half (odd/even games) coverage ≥ **0.95** under the posterior SE (obs ≥ 0.9995) and in **[0.92, 0.98]** under the sampling SE for O/D/net (obs 0.9465–0.9601, nominal 0.954) | 2026-09-01 (SE columns; coefficients unchanged) | annual cron + dispatch (`.github/workflows/ncaa_mbb_models.yml`) via `python/ncaa_mbb_model_01_rapm_league.py` |
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

## Adopted 2026-09-02: multi-year pooling (PR to `feat/rapm-default-on`)

The multi-year lever measured in PR #21 is now the producer default for every season
with a full 3-season window (2013+). Nothing about the measurement changed; what changed
is that the two blockers recorded then are settled:

1. **The σ̂² band was re-derived, not widened.** The blocker as filed said a pooled fit
   measures **7,733** against a band of [11000, 15000]. That number was an artifact:
   `sigma2 = RSS_w / (rows − df_eff)` divides a weight-deflated numerator by an
   undeflated row count, so a decayed pool reports it low by exactly `mean(decay)` —
   0.583 on a 3-season 0.5-decay pool, and 7,733/13,212 = 0.585. sdv-py PR #441 divides
   by `sum(fit_weight) − df_eff` instead, which is the same number whenever `fit_weight`
   is absent, so no published value moves. The band was then re-derived from a **14-season
   `--all --survey` sweep of the pooled estimator** (every gate computed, none of them
   5(a)): observed **12,646.5 (2014) … 13,361.0 (2026)** → **[11000, 15000]** by the
   ±12.5%-and-round-outward rule, which reproduces both 2026-09-01 flat bands exactly
   from their own extremes. The pooled band comes out equal to the flat one *because the
   statistic means the same thing again*, not because one was stretched to fit the other.
2. **The published frame is filtered to season *t*.** `season_slice` (sdv-py) inner-joins
   the pooled players to the target season's own exposure, which drops the window-only
   players and restores season-*t* `off_poss` / `def_poss` in one step.
   `tests/test_rapm_leakage_boundary.py` proves both halves: every `_data_file` read of a
   real season is instrumented and no season after the target is touched, and a pooled
   fit that rates a prior-season-only player publishes a frame without him, on the target
   season's possessions rather than the two-season sum.

Torvik team-aggregate Spearman **improves** on every pooled season (e.g. 2024
0.9725 → 0.9796, 2021 0.9434 → 0.9646); no gate floor was touched.

## Evaluated, NOT adopted: the SPM prior (2026-09-02)

The SPM prior wins every point-estimate criterion (0.061 pts/game of out-of-sample margin
error, next-season Spearman 0.3115 → 0.3466) and **fails gate 5(d)**, the sampling-SE
split-half calibration frozen 2026-09-01 with band [0.92, 0.98]. Measured per lever by
`ops/experiments/rapm_se_calibration.py` (orapm / drapm / rapm_net):

| season | flat | pooled | SPM only | pooled + SPM |
|---|---|---|---|---|
| mbb 2024 | 0.9465 0.9561 0.9517 | 0.9604 0.9724 0.9660 | **0.7979 0.9136 0.8259** | **0.8419 0.9375 0.8650** |
| mbb 2019 | 0.9545 0.9536 0.9542 | 0.9674 0.9688 0.9634 | **0.8152 0.9187 0.8473** | **0.8525 0.9369 0.8798** |

The cause is structural: `solve_rapm_league` treats the prior mean `b0` as a fixed
constant, so the published SE describes `beta − b0` only, while `b0` is itself estimated
from the season's box scores and moves under a refit. z-sd is 1.39–1.53 against a nominal
1.0 — the shipped intervals would be ~35% too narrow. The gate was not lowered.
Propagating `Var(b0)` into the posterior covariance is the fix and is a modelling task.

**Finding worth keeping:** the multi-year gain is **not** concentrated below ~200
possessions, which is what the original backlog item assumed. The absolute next-season
Spearman gain is flat across playing time (+0.022 in the <100 bin and +0.022 in the 1500+
bin), so there is no threshold above which it stops helping — it is a uniform variance
reduction, not a tail stabiliser. The *relative* gain is largest in the tail only because
the baseline is near zero there; those are different claims.

## Operability (Track C steps 2–6)

- `models/manifest.yaml` — single home for the model/stage list (guarded by `tests/test_model_manifest.py`).
- One estimand = one numbered pipeline, flat in `python/` beside the data stages: `ncaa_mbb_model_01_rapm_league.py` / `ncaa_mbb_model_02_rapm_within_team.py`; run subsets with `scripts/ncaa_mbb_models.sh`.
- League-wide retrain is now WIRED: `.github/workflows/ncaa_mbb_models.yml` (dispatch + annual post-season cron). Within-team stays manual by design (raw HTML bundle dependency).
- Fingerprint skip: deliberately NOT used — inputs are living published trees; every run recomputes. Each run appends `models/ledger.jsonl`.
- Step 6: per-season parquet assets live on the release tags; nothing fitted is committed here.
