"""Direct family extractor -- NCAA's parsed families are already tidy.

Unlike WNBA (which reshapes raw ESPN JSON through per-family helpers), the
NCAA parser already emits tidy per-family row lists. The only job here is to
build a frame from those rows and pin ``contest_id``/``season`` onto it.
"""

from __future__ import annotations

from typing import Any

import polars as pl


def _dig(d: Any, *keys: str) -> Any:
    """None-safe nested dict lookup: ``_dig(r, "a", "b")`` == ``r["a"]["b"]`` or None."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _player_names(entries: Any) -> "list[str]":
    if not isinstance(entries, list):
        return []
    return [n for n in (_dig(p, "id", "name") for p in entries) if n]


def _flatten_lineup_row(r: dict) -> "dict[str, Any]":
    """One nested engine lineup-stint row -> flat scalars.

    The engine emits deep structs (score_info, team/opponent, players lists,
    team_stats/opponent_stats shot-family trees) whose exact shape varies
    game-to-game -- flattening the DICT here is robust to absent branches
    (``.get`` chains) where polars ``struct.field`` on a Null-typed field
    errors, and it removes the cross-game struct-schema concat hazard.
    Stats flatten at the TOTALS level per side (``""``/``opp_`` prefixes).
    """
    # ponytail: totals only -- the early/orb sub-splits stay in the raw JSON;
    # add columns here if a consumer ever needs them.
    players = _player_names(r.get("players"))
    out: "dict[str, Any]" = {
        "date": r.get("date"),
        "location_type": r.get("location_type"),
        "team": _dig(r, "team", "team", "name"),
        "team_year": _dig(r, "team", "year", "value"),
        "opponent": _dig(r, "opponent", "team", "name"),
        "lineup_id": _dig(r, "lineup_id", "value"),
        "start_min": r.get("start_min"),
        "end_min": r.get("end_min"),
        "duration_mins": r.get("duration_mins"),
        **{
            f"player_{i + 1}": (players[i] if i < len(players) else None)
            for i in range(5)
        },
        "players_in": " | ".join(_player_names(r.get("players_in"))) or None,
        "players_out": " | ".join(_player_names(r.get("players_out"))) or None,
        "start_scored": _dig(r, "score_info", "start", "scored"),
        "start_allowed": _dig(r, "score_info", "start", "allowed"),
        "end_scored": _dig(r, "score_info", "end", "scored"),
        "end_allowed": _dig(r, "score_info", "end", "allowed"),
        "start_diff": _dig(r, "score_info", "start_diff"),
        "end_diff": _dig(r, "score_info", "end_diff"),
        "player_count_error": r.get("player_count_error"),
    }
    for prefix, side in (("", "team_stats"), ("opp_", "opponent_stats")):
        s = r.get(side) or {}
        g = lambda *ks: _dig(s, *ks)  # noqa: E731 - tiny local accessor
        out.update(
            {
                f"{prefix}poss": g("num_possessions"),
                f"{prefix}pts": g("pts"),
                f"{prefix}plus_minus": g("plus_minus"),
                f"{prefix}fga": g("fg", "attempts", "total"),
                f"{prefix}fgm": g("fg", "made", "total"),
                f"{prefix}rima": g("fg_rim", "attempts", "total"),
                f"{prefix}rimm": g("fg_rim", "made", "total"),
                f"{prefix}rim_ast": g("fg_rim", "ast", "total"),
                f"{prefix}mida": g("fg_mid", "attempts", "total"),
                f"{prefix}midm": g("fg_mid", "made", "total"),
                f"{prefix}mid_ast": g("fg_mid", "ast", "total"),
                f"{prefix}fg2a": g("fg_2p", "attempts", "total"),
                f"{prefix}fg2m": g("fg_2p", "made", "total"),
                f"{prefix}tpa": g("fg_3p", "attempts", "total"),
                f"{prefix}tpm": g("fg_3p", "made", "total"),
                f"{prefix}tp_ast": g("fg_3p", "ast", "total"),
                f"{prefix}fta": g("ft", "attempts", "total"),
                f"{prefix}ftm": g("ft", "made", "total"),
                f"{prefix}orb": g("orb", "total"),
                f"{prefix}drb": g("drb", "total"),
                f"{prefix}to": g("to", "total"),
                f"{prefix}stl": g("stl", "total"),
                f"{prefix}blk": g("blk", "total"),
                f"{prefix}ast": g("assist", "total"),
                f"{prefix}foul": g("foul", "total"),
            }
        )
    return out


def extract_family(
    final: dict, family: str, *, season: int, contest_id: str
) -> pl.DataFrame:
    """Build the family's frame from one game's parsed JSON, tagged with contest_id/season.

    ``contest_id`` always overwrites any value already present in the rows,
    pinning it to Utf8 (dtype discipline: contest_id is Utf8 everywhere).
    An empty family still returns the two literal columns so
    ``pl.concat(..., how="diagonal_relaxed")`` across games works even when
    some games have an empty family.
    """
    rows = final.get(family) or []
    if family == "lineups":
        rows = [_flatten_lineup_row(r) for r in rows]
    if not rows:
        # pl.DataFrame([]) is 0x0 -- with_columns would broadcast the
        # literals to 1 row instead of 0. Build the empty frame with an
        # explicit 0-row schema so it stays concat-safe.
        return pl.DataFrame(schema={"contest_id": pl.Utf8, "season": pl.Int64})
    frame = pl.DataFrame(rows)
    return frame.with_columns(
        pl.lit(contest_id, dtype=pl.Utf8).alias("contest_id"),
        pl.lit(season, dtype=pl.Int64).alias("season"),
    )
