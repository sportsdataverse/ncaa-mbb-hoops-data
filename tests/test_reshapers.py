"""Tests for reshapers.py -- direct family extractor, hermetic (real fixture, no network)."""

import json
from pathlib import Path

import polars as pl

from ncaa_mbb_data_build.reshapers import extract_family

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "raw_root"
    / "mbb"
    / "json"
    / "1613299.json"
)

FAMILY_HEIGHTS = {
    "pbp": 502,
    "lineups": 58,
    "player_box": 20,
    "team_box": 2,
    "shots": 104,
    "possessions": 139,
}


def _load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_extract_family_all_six_families_from_real_fixture():
    final = _load_fixture()
    for fam, expected_height in FAMILY_HEIGHTS.items():
        df = extract_family(final, fam, season=2026, contest_id="1613299")
        assert df.height == expected_height, (
            f"{fam}: expected {expected_height}, got {df.height}"
        )
        assert df.height > 0
        assert "contest_id" in df.columns
        assert "season" in df.columns
        assert df.schema["contest_id"] == pl.Utf8
        assert (df.get_column("contest_id") == "1613299").all()
        assert (df.get_column("season") == 2026).all()


def test_extract_family_empty_family_is_concat_safe():
    df = extract_family({"pbp": []}, "pbp", season=2026, contest_id="X")

    assert df.height == 0
    assert "contest_id" in df.columns
    assert "season" in df.columns
    assert df.schema["contest_id"] == pl.Utf8


def test_extract_family_empty_and_nonempty_concat_diagonal_relaxed():
    final = _load_fixture()
    non_empty = extract_family(final, "pbp", season=2026, contest_id="1613299")
    empty = extract_family({"pbp": []}, "pbp", season=2026, contest_id="X")

    out = pl.concat([empty, non_empty], how="diagonal_relaxed")

    assert out.height == non_empty.height


def test_lineups_flatten_real_shape():
    """Nested engine lineup row -> all-scalar flat row; absent branches -> None."""
    from ncaa_mbb_data_build.reshapers import _flatten_lineup_row, extract_family

    row = {
        "date": "2025-11-03T17:00:00",
        "location_type": "HOME",
        "start_min": 0.0,
        "end_min": 4.3,
        "duration_mins": 4.3,
        "score_info": {
            "start": {"scored": 0, "allowed": 0},
            "end": {"scored": 8, "allowed": 6},
            "start_diff": 0,
            "end_diff": 2,
        },
        "team": {"team": {"name": "Buffalo"}, "year": {"value": 2025}},
        "opponent": {"team": {"name": "Southern Miss."}, "year": {"value": 2025}},
        "lineup_id": {"value": "abc"},
        "players": [
            {"code": "p1", "id": {"name": "Brizzi, Angelo"}, "ncaa_id": None},
            {"code": "p2", "id": {"name": "Freitag, Daniel"}, "ncaa_id": None},
            {"code": "p3", "id": {"name": "McKenna, Ezra"}, "ncaa_id": None},
            {"code": "p4", "id": {"name": "Jones, Kyle"}, "ncaa_id": None},
            {"code": "p5", "id": {"name": "Batchelor, Noah"}, "ncaa_id": None},
        ],
        "players_in": [],
        "players_out": [],
        "team_stats": {
            "num_events": 10,
            "num_possessions": 8,
            "pts": 8,
            "plus_minus": 2,
            "fg": {"attempts": {"total": 7}, "made": {"total": 3}, "ast": None},
            "fg_rim": {"attempts": {"total": 3}, "made": {"total": 2}, "ast": {"total": 1}},
            "orb": {"total": 1},
            "to": {"total": 2},
        },
        # opponent_stats ABSENT on purpose -- absent branch must yield Nones
        "player_count_error": None,
    }
    flat = _flatten_lineup_row(row)
    assert all(not isinstance(v, (dict, list)) for v in flat.values())
    assert flat["team"] == "Buffalo" and flat["opponent"] == "Southern Miss."
    assert flat["player_1"] == "Batchelor, Noah" and flat["player_5"] == "McKenna, Ezra"  # sorted
    assert flat["pts"] == 8 and flat["fga"] == 7 and flat["rim_ast"] == 1
    assert flat["opp_pts"] is None and flat["opp_fga"] is None
    assert flat["players_in"] is None  # empty list -> None, not ""

    # end-to-end through extract_family: flat frame, no Struct/List dtypes
    df = extract_family({"lineups": [row]}, "lineups", season=2026, contest_id="6388769")
    assert df.height == 1
    assert not any(isinstance(t, (type(None),)) for t in df.dtypes)  # sanity
    assert all(t.base_type() not in (pl.Struct, pl.List) for t in df.dtypes)
    assert df.get_column("contest_id").to_list() == ["6388769"]
