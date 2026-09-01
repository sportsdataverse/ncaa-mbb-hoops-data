# NCAA MBB RAPM — league-wide and within-team


Two deliberately different RAPM estimands publish from this repository
on separate release tags (never merged): **league-wide**
(`ncaa_mbb_rapm`, Path B — every D-I player on one
points-per-100-possessions scale) and **within-team**
(`ncaa_mbb_rapm_within_team`, Path A — how one team’s performance
apportions across its own players; not comparable across teams by
construction). Every published row carries an `estimand` column so the
two can never be silently conflated. This document is the reproducible
writeup for the league-wide model, computed at render time from the
local sweep outputs (`ops/out_league/`), the committed evaluation card,
and the committed Torvik oracle fixture.

**Model.** Ridge regression (λ = 1000) over the possession-level on/off
design matrix: each row is a possession, each player an on/off column
(+1 offense, −1 defense), plus intercept and home-court terms. Offensive
and defensive components are estimated jointly; `rapm = orapm + drapm`
on the points-per-100 scale. The regularization strength was fixed by
the 2026-08 validation program — and the fitted value is *asserted* at
run time, because the λ no-op incident (a ridge that silently fit
unregularized) is exactly the class of bug that survives rank-based
checks.

**Feature engineering lives in the possession construction, not the
regression**: substitution-window stint building, possession trimming
(the usable-possession gate measures how much of a season survives),
free-throw and end-of-period edge handling, and the `name_changes`
identity bridge that keeps a player one column across transfers and name
variants. The intercept and home-court estimates double as **scale-bug
catchers** — a wrong points-per-possession normalization moves them out
of band even when the player ordering (what a Spearman gate sees)
survives.

## Training data

<div id="qozolhgtoq" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#qozolhgtoq table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#qozolhgtoq thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#qozolhgtoq p { margin: 0; padding: 0; }
 #qozolhgtoq .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #qozolhgtoq .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #qozolhgtoq .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #qozolhgtoq .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #qozolhgtoq .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #qozolhgtoq .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #qozolhgtoq .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #qozolhgtoq .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #qozolhgtoq .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #qozolhgtoq .gt_column_spanner_outer:first-child { padding-left: 0; }
 #qozolhgtoq .gt_column_spanner_outer:last-child { padding-right: 0; }
 #qozolhgtoq .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #qozolhgtoq .gt_spanner_row { border-bottom-style: hidden; }
 #qozolhgtoq .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #qozolhgtoq .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #qozolhgtoq .gt_from_md> :first-child { margin-top: 0; }
 #qozolhgtoq .gt_from_md> :last-child { margin-bottom: 0; }
 #qozolhgtoq .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #qozolhgtoq .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #qozolhgtoq .gt_indent_1 { text-indent: 5px; }
 #qozolhgtoq .gt_indent_2 { text-indent: calc(5px * 2); }
 #qozolhgtoq .gt_indent_3 { text-indent: calc(5px * 3); }
 #qozolhgtoq .gt_indent_4 { text-indent: calc(5px * 4); }
 #qozolhgtoq .gt_indent_5 { text-indent: calc(5px * 5); }
 #qozolhgtoq .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #qozolhgtoq .gt_row_group_first td { border-top-width: 2px; }
 #qozolhgtoq .gt_row_group_first th { border-top-width: 2px; }
 #qozolhgtoq .gt_striped { color: #333333; background-color: #F4F4F4; }
 #qozolhgtoq .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #qozolhgtoq .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #qozolhgtoq .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #qozolhgtoq .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #qozolhgtoq .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #qozolhgtoq .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #qozolhgtoq .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #qozolhgtoq .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #qozolhgtoq .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #qozolhgtoq .gt_left { text-align: left; }
 #qozolhgtoq .gt_center { text-align: center; }
 #qozolhgtoq .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #qozolhgtoq .gt_font_normal { font-weight: normal; }
 #qozolhgtoq .gt_font_bold { font-weight: bold; }
 #qozolhgtoq .gt_font_italic { font-style: italic; }
 #qozolhgtoq .gt_super { font-size: 65%; }
 #qozolhgtoq .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #qozolhgtoq .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #qozolhgtoq .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #qozolhgtoq .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #qozolhgtoq .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #qozolhgtoq .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| League-wide RAPM corpus, by season |  |  |  |  |  |
|----|----|----|----|----|----|
| players + possessions from the local sweep outputs; usable% / intercept / HCA from the frozen evaluation card |  |  |  |  |  |
| season | players | player_poss | usable | mu | hca |
| 2011 | 4482 | 5,994,840 | 77.38 | 100.24 | 2.81 |
| 2012 | 4537 | 6,170,570 | 79.90 | 99.98 | 3.00 |
| 2013 | 4542 | 5,887,870 | 75.38 | 99.24 | 2.93 |
| 2014 | 4671 | 6,244,830 | 77.14 | 103.00 | 2.64 |
| 2015 | 4649 | 6,133,910 | 78.09 | 100.85 | 2.82 |
| 2016 | 4640 | 6,284,410 | 75.07 | 102.54 | 2.82 |
| 2017 | 4630 | 6,391,860 | 75.58 | 102.45 | 2.50 |
| 2018 | 4592 | 6,286,390 | 74.14 | 103.17 | 2.67 |
| 2019 | 4688 | 6,669,340 | 78.87 | 102.30 | 2.54 |
| 2020 | 4626 | 6,776,490 | 83.05 | 99.98 | 2.79 |
| 2021 | 4896 | 5,091,020 | 84.36 | 100.88 | 2.02 |
| 2022 | 4942 | 7,197,890 | 86.70 | 101.54 | 2.19 |
| 2023 | 4942 | 7,408,220 | 85.83 | 102.61 | 2.74 |
| 2024 | 4919 | 7,402,980 | 84.87 | 104.84 | 2.46 |
| 2025 | 4960 | 7,275,190 | 83.30 | 105.67 | 2.69 |
| 2026 | 4974 | 7,889,700 | 89.84 | 107.97 | 2.44 |

&#10;</div>

Inputs are this repository’s own published trees: `possessions` (the
stint-level frame the NCAA hoops engine compiles from raw stats.ncaa.org
bundles), `team_rosters`, and `name_changes`. Seasons 2011–2026; 2010
exists upstream but fails the usable-possession gate and is excluded by
the gate rather than a hand list.

## Exploratory data analysis

<img src="rapm_files/figure-commonmark/cell-4-output-1.png" width="420"
height="300"
alt="League-wide RAPM distribution, latest season — ridge shrinkage centers the mass at zero." />

<img src="rapm_files/figure-commonmark/cell-5-output-1.png" width="420"
height="300"
alt="Offensive vs defensive RAPM, latest season — the two components are estimated jointly." />

<img src="rapm_files/figure-commonmark/cell-6-output-1.png" width="420"
height="300"
alt="Shrinkage in action: |RAPM| vs possessions played — low-minute players are pulled to zero." />

## Attribution

There is no SHAP section here, and deliberately so: in an on/off design
the “features” are the players themselves, so the fitted coefficients
*are* the attributions — each player’s RAPM is exactly their estimated
marginal effect on a possession’s outcome, jointly with everyone else’s.
The O/D scatter above is the model’s native attribution decomposition,
and the shrinkage plot shows the prior doing the work SHAP would
otherwise reveal: low-minute players carry near-zero attributed effect
because the data cannot separate them from their lineups.

## Evaluation

Two layers, both real. First, the **frozen gates from the 2026-08-24
full validation sweep** — publish-blocking (a failed season writes
NOTHING; floors may only be raised):

| gate | floor | observed (sweep) |
|----|----|----|
| usable-possession fraction | ≥ 0.65 | 2011+ min 0.7414 |
| intercept era band (scale-bug catcher) | \[95, 112\] | 99.24–107.97 |
| home-court advantage band | \[1.0, 4.0\] | in-band all seasons |
| Torvik external (league-wide only) | ≥ 250 joined teams AND Spearman(team_net, adjem) ≥ 0.93 | min 0.9434 |

<div id="cabaobtilz" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#cabaobtilz table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#cabaobtilz thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#cabaobtilz p { margin: 0; padding: 0; }
 #cabaobtilz .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #cabaobtilz .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #cabaobtilz .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #cabaobtilz .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #cabaobtilz .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #cabaobtilz .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #cabaobtilz .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #cabaobtilz .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #cabaobtilz .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #cabaobtilz .gt_column_spanner_outer:first-child { padding-left: 0; }
 #cabaobtilz .gt_column_spanner_outer:last-child { padding-right: 0; }
 #cabaobtilz .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #cabaobtilz .gt_spanner_row { border-bottom-style: hidden; }
 #cabaobtilz .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #cabaobtilz .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #cabaobtilz .gt_from_md> :first-child { margin-top: 0; }
 #cabaobtilz .gt_from_md> :last-child { margin-bottom: 0; }
 #cabaobtilz .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #cabaobtilz .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #cabaobtilz .gt_indent_1 { text-indent: 5px; }
 #cabaobtilz .gt_indent_2 { text-indent: calc(5px * 2); }
 #cabaobtilz .gt_indent_3 { text-indent: calc(5px * 3); }
 #cabaobtilz .gt_indent_4 { text-indent: calc(5px * 4); }
 #cabaobtilz .gt_indent_5 { text-indent: calc(5px * 5); }
 #cabaobtilz .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #cabaobtilz .gt_row_group_first td { border-top-width: 2px; }
 #cabaobtilz .gt_row_group_first th { border-top-width: 2px; }
 #cabaobtilz .gt_striped { color: #333333; background-color: #F4F4F4; }
 #cabaobtilz .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #cabaobtilz .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #cabaobtilz .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #cabaobtilz .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #cabaobtilz .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #cabaobtilz .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #cabaobtilz .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #cabaobtilz .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #cabaobtilz .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #cabaobtilz .gt_left { text-align: left; }
 #cabaobtilz .gt_center { text-align: center; }
 #cabaobtilz .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #cabaobtilz .gt_font_normal { font-weight: normal; }
 #cabaobtilz .gt_font_bold { font-weight: bold; }
 #cabaobtilz .gt_font_italic { font-style: italic; }
 #cabaobtilz .gt_super { font-size: 65%; }
 #cabaobtilz .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #cabaobtilz .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #cabaobtilz .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #cabaobtilz .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #cabaobtilz .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #cabaobtilz .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Per-season gate results — frozen from the validation sweep (evaluation card) |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
| rho = Spearman(team_net, Torvik AdjEM) on n joined teams; the possession-weighted stint aggregate the gate uses |  |  |  |  |  |  |
| season | usable | players | mu | hca | rho | n |
| 2011 | 77.38 | 4482 | 100.24 | 2.81 | 0.9598 | 332 |
| 2012 | 79.90 | 4537 | 99.98 | 3.00 | 0.9706 | 333 |
| 2013 | 75.38 | 4542 | 99.24 | 2.93 | 0.9591 | 334 |
| 2014 | 77.14 | 4671 | 103.00 | 2.64 | 0.9584 | 339 |
| 2015 | 78.09 | 4649 | 100.85 | 2.82 | 0.9617 | 339 |
| 2016 | 75.07 | 4640 | 102.54 | 2.82 | 0.9587 | 339 |
| 2017 | 75.58 | 4630 | 102.45 | 2.50 | 0.9653 | 338 |
| 2018 | 74.14 | 4592 | 103.17 | 2.67 | 0.9509 | 338 |
| 2019 | 78.87 | 4688 | 102.30 | 2.54 | 0.9719 | 338 |
| 2020 | 83.05 | 4626 | 99.98 | 2.79 | 0.9711 | 337 |
| 2021 | 84.36 | 4896 | 100.88 | 2.02 | 0.9434 | 335 |
| 2022 | 86.70 | 4942 | 101.54 | 2.19 | 0.9770 | 346 |
| 2023 | 85.83 | 4942 | 102.61 | 2.74 | 0.9714 | 350 |
| 2024 | 84.87 | 4919 | 104.84 | 2.46 | 0.9725 | 350 |
| 2025 | 83.30 | 4960 | 105.67 | 2.69 | 0.9652 | 353 |
| 2026 | 89.84 | 4974 | 107.97 | 2.44 | 0.9783 | 356 |

&#10;</div>

Second, a **render-time external check** against the committed Torvik
oracle fixture. The gate’s own `team_net` is a possession-weighted stint
aggregate that needs the full stint frame; this document computes the
closest player-level proxy — each team’s possession-weighted mean of
player RAPM — and holds it against Torvik AdjEM. It is labeled a proxy
precisely because it is not the gate’s aggregate; its job is to show the
external agreement reproduces from the committed artifacts alone:

<div id="evxktmmmhl" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#evxktmmmhl table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#evxktmmmhl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#evxktmmmhl p { margin: 0; padding: 0; }
 #evxktmmmhl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #evxktmmmhl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #evxktmmmhl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #evxktmmmhl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #evxktmmmhl .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #evxktmmmhl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #evxktmmmhl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #evxktmmmhl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #evxktmmmhl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #evxktmmmhl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #evxktmmmhl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #evxktmmmhl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #evxktmmmhl .gt_spanner_row { border-bottom-style: hidden; }
 #evxktmmmhl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #evxktmmmhl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #evxktmmmhl .gt_from_md> :first-child { margin-top: 0; }
 #evxktmmmhl .gt_from_md> :last-child { margin-bottom: 0; }
 #evxktmmmhl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #evxktmmmhl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #evxktmmmhl .gt_indent_1 { text-indent: 5px; }
 #evxktmmmhl .gt_indent_2 { text-indent: calc(5px * 2); }
 #evxktmmmhl .gt_indent_3 { text-indent: calc(5px * 3); }
 #evxktmmmhl .gt_indent_4 { text-indent: calc(5px * 4); }
 #evxktmmmhl .gt_indent_5 { text-indent: calc(5px * 5); }
 #evxktmmmhl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #evxktmmmhl .gt_row_group_first td { border-top-width: 2px; }
 #evxktmmmhl .gt_row_group_first th { border-top-width: 2px; }
 #evxktmmmhl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #evxktmmmhl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #evxktmmmhl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #evxktmmmhl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #evxktmmmhl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #evxktmmmhl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #evxktmmmhl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #evxktmmmhl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #evxktmmmhl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #evxktmmmhl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #evxktmmmhl .gt_left { text-align: left; }
 #evxktmmmhl .gt_center { text-align: center; }
 #evxktmmmhl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #evxktmmmhl .gt_font_normal { font-weight: normal; }
 #evxktmmmhl .gt_font_bold { font-weight: bold; }
 #evxktmmmhl .gt_font_italic { font-style: italic; }
 #evxktmmmhl .gt_super { font-size: 65%; }
 #evxktmmmhl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #evxktmmmhl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #evxktmmmhl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #evxktmmmhl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #evxktmmmhl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #evxktmmmhl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Render-time external check — possession-weighted player-RAPM proxy vs Torvik AdjEM |  |  |
|----|----|----|
| proxy aggregate (not the gate's stint-weighted team_net); computed from committed artifacts on every render |  |  |
| season | joined_teams | proxy_spearman |
| 2011 | 332 | 0.9596 |
| 2012 | 333 | 0.9706 |
| 2013 | 334 | 0.9591 |
| 2014 | 339 | 0.9581 |
| 2015 | 339 | 0.9616 |
| 2016 | 339 | 0.9590 |
| 2017 | 338 | 0.9653 |
| 2018 | 338 | 0.9510 |
| 2019 | 338 | 0.9720 |
| 2020 | 337 | 0.9711 |
| 2021 | 335 | 0.9434 |
| 2022 | 346 | 0.9770 |
| 2023 | 350 | 0.9715 |
| 2024 | 350 | 0.9726 |
| 2025 | 353 | 0.9654 |
| 2026 | 356 | 0.9783 |

&#10;</div>

<img src="rapm_files/figure-commonmark/cell-9-output-1.png" width="420"
height="300"
alt="Team proxy aggregate vs Torvik AdjEM, latest season." />

## Results

<div id="bcljltknzf" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#bcljltknzf table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#bcljltknzf thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#bcljltknzf p { margin: 0; padding: 0; }
 #bcljltknzf .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #bcljltknzf .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #bcljltknzf .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #bcljltknzf .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #bcljltknzf .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #bcljltknzf .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #bcljltknzf .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #bcljltknzf .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #bcljltknzf .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #bcljltknzf .gt_column_spanner_outer:first-child { padding-left: 0; }
 #bcljltknzf .gt_column_spanner_outer:last-child { padding-right: 0; }
 #bcljltknzf .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #bcljltknzf .gt_spanner_row { border-bottom-style: hidden; }
 #bcljltknzf .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #bcljltknzf .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #bcljltknzf .gt_from_md> :first-child { margin-top: 0; }
 #bcljltknzf .gt_from_md> :last-child { margin-bottom: 0; }
 #bcljltknzf .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #bcljltknzf .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #bcljltknzf .gt_indent_1 { text-indent: 5px; }
 #bcljltknzf .gt_indent_2 { text-indent: calc(5px * 2); }
 #bcljltknzf .gt_indent_3 { text-indent: calc(5px * 3); }
 #bcljltknzf .gt_indent_4 { text-indent: calc(5px * 4); }
 #bcljltknzf .gt_indent_5 { text-indent: calc(5px * 5); }
 #bcljltknzf .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #bcljltknzf .gt_row_group_first td { border-top-width: 2px; }
 #bcljltknzf .gt_row_group_first th { border-top-width: 2px; }
 #bcljltknzf .gt_striped { color: #333333; background-color: #F4F4F4; }
 #bcljltknzf .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #bcljltknzf .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #bcljltknzf .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #bcljltknzf .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #bcljltknzf .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #bcljltknzf .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #bcljltknzf .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #bcljltknzf .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #bcljltknzf .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #bcljltknzf .gt_left { text-align: left; }
 #bcljltknzf .gt_center { text-align: center; }
 #bcljltknzf .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #bcljltknzf .gt_font_normal { font-weight: normal; }
 #bcljltknzf .gt_font_bold { font-weight: bold; }
 #bcljltknzf .gt_font_italic { font-style: italic; }
 #bcljltknzf .gt_super { font-size: 65%; }
 #bcljltknzf .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #bcljltknzf .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #bcljltknzf .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #bcljltknzf .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #bcljltknzf .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #bcljltknzf .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Top 15 league-wide RAPM — 2026 (min 800 possessions) |  |  |  |  |  |
|----|----|----|----|----|----|
| points per 100 possessions; no public headshot CDN exists for stats.ncaa.org player ids |  |  |  |  |  |
| Player | Team | Poss | O-RAPM | D-RAPM | RAPM |
| YAXEL.LENDEBORG | Michigan | 4,259 | 8.70 | 6.96 | 15.66 |
| FLORY.BIDUNGA | Kansas | 3,724 | 5.10 | 7.97 | 13.07 |
| HENRI.VEESAAR | North Carolina | 3,342 | 6.24 | 6.47 | 12.70 |
| MJ.COLLINS | Utah St. | 3,758 | 6.84 | 4.94 | 11.78 |
| JAKOBI.GILLESPIE | Tennessee | 4,360 | 9.17 | 2.08 | 11.25 |
| FLETCHER.LOYER | Purdue | 3,758 | 5.96 | 4.64 | 10.60 |
| KEATON.WAGLER | Illinois | 3,979 | 7.65 | 2.71 | 10.37 |
| JEREMY.FEARS | Michigan St. | 3,852 | 4.21 | 5.65 | 9.86 |
| ROBBIE.AVILA | Saint Louis | 3,295 | 6.02 | 3.68 | 9.71 |
| JOSHUA.JEFFERSON | Iowa St. | 3,702 | 5.80 | 3.81 | 9.60 |
| DAME.SARR | Duke | 2,957 | 3.22 | 6.31 | 9.53 |
| IZAIYAH.NELSON | South Fla. | 3,323 | 4.23 | 5.28 | 9.51 |
| MILAN.MOMCILOVIC | Iowa St. | 3,832 | 3.79 | 5.67 | 9.46 |
| TREY.MCKENNEY | Michigan | 3,135 | 4.25 | 5.12 | 9.38 |
| CAMERON.BOOZER | Duke | 4,187 | 6.46 | 2.90 | 9.36 |

&#10;</div>

RAPM is a retrodictive on/off estimate: low-minute players shrink
heavily toward zero, multicollinearity between players who always share
the floor is resolved only by the prior, and the external Torvik gate
validates TEAM-level aggregation, not individual ordering — which is why
the results table applies a possession floor before ranking anyone.

## Provenance & reproducibility

- **Trained on:** this repository’s published `possessions` +
  `team_rosters` + `name_changes` trees, seasons 2011–2026 (2010
  excluded by the usable-possession gate).
- **Model:** ridge (λ = 1000, asserted at run time) via the sdv-py
  `mbb_ncaa_rapm_league` engine; league-wide stage
  `python/ncaa_mbb_model_01_rapm_league.py`, within-team stage
  `python/ncaa_mbb_model_02_rapm_within_team.py` (manual by design —
  needs the raw HTML bundle checkout).
- **Gates:** frozen in the table above; oracle fixture
  `ops/oracle/ncaa_mbb_torvik.parquet` (a NaN rho or missing oracle
  season is a FAILURE, never a skip). Runs append `models/ledger.jsonl`;
  publish is a separate deliberate step (`ops/publish_rapm_league.py`).
- **Retrain:** `scripts/ncaa_mbb_models.sh 01` /
  `.github/workflows/ncaa_mbb_models.yml` (dispatch + annual post-season
  cron). Single home: `models/manifest.yaml`.
- **Rebuild this document:** `scripts/render_model_docs.sh` (Quarto →
  GFM; `uv sync --group docs`); reads only committed/local artifacts —
  fully offline.

## Avenues for improvement & open issues

- **Luck adjustment and archetype priors** — the two known gaps versus
  the strongest public APM systems (catalogued in the APM research
  corpus): 3P% luck-adjusting the target, and informative priors by
  player archetype instead of a flat ridge.
- **Exact standard errors** — the ridge posterior gives per-player SEs
  almost for free; publishing them would turn point estimates into
  honest intervals.
- **Within-team CI** — Path A still requires the raw HTML bundle
  checkout; a store-backed runner would let it join the wired retrain.
- **Known issue:** multi-year RAPM (stabilizing low-minute players
  across seasons) is unbuilt; single-season estimates stay noisy below
  ~200 possessions.
