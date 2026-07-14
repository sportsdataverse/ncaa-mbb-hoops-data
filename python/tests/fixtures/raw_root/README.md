# Hermetic raw-root fixture (8 NCAA MBB games)

Mirrors the `ncaa-mbb-hoops-raw` on-disk tree so `ingest.read_parsed` /
`ingest.season_contest_ids` and the build/e2e tests run fully offline.

- `mbb/json/{contest_id}.json` — the 7-key parsed dict
  (`{contest_id, pbp, lineups, player_box, team_box, shots, possessions}`)
  for 8 games, produced by the raw repo's `ncaa_parse.parse_bundle(league="mbb")`
  over the committed bigballR HTML fixtures at
  `sdv-py/tests/fixtures/ncaa/bigballr/html/{pbp,box,individual_stats}_{cid}.html`.
- `mbb/schedule_master.parquet` — `contest_id` (Utf8), `season` (Utf8, `"2026"`),
  `captured` (Utf8); one row per game. Season label is synthetic (all 2026) so
  `season_contest_ids(2026)` returns all 8 — the games span real 2024-25/2025-26
  dates but season is only a filter key here.

Contest ids: 1613299, 5722355, 5728709, 5732292, 5733807, 6470186, 6479592, 6479639.

Regenerate: re-run `parse_bundle` over the HTML fixtures (sdv-py **main** venv —
PyPI `sportsdataverse` lacks the NCAA parsers) and rewrite these files. The
parsed contract is stable; column drift in sdv-py's NCAA parsers is what would
change them.
