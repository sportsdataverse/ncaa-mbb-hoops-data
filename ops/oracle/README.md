# Torvik team-rating oracle (league-RAPM external gate)

`ncaa_mbb_torvik.parquet` — Bart Torvik men's T-Rank adjusted efficiencies,
2010–2026 (5,823 team-seasons), crosswalked to stats.ncaa.org team names.

- Source: `https://barttorvik.com/<year>_team_results.csv` (the static CSV;
  `trank.php?csv=1` returns HTTP-200 HTML and must not be used).
- Captured 2026-08-18 by sdv-py `dev/ncaa_rapm/build_oracle.py` (greedy
  claim-consuming name matcher, ≈97% of D-I matched; unmatched tail is
  paren-disambiguated one-offs like `Saint Mary's (CA)`, deliberately not
  fuzzy-matched). Regenerate there and re-copy.
- Columns: `season` (**Utf8**), `team` (stats.ncaa.org name — the join key),
  `torvik_team`, `adjoe`, `adjde`, `barthag`, `adjem` (= adjoe − adjde).
  No null or imputed-zero efficiency rows (verified 2026-08-24).

Consumed by `ops/build_rapm_league.py` gate 3: per-season
Spearman(team_net, adjem) ≥ 0.93 on ≥ 250 joined teams (floors frozen from
the observed 2026-08-24 sweep — min 0.9434, median 0.9653; never lowered).
