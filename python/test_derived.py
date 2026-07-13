"""Tests for derived.py -- the 3 non-family datasets, hermetic (real fixtures, no network)."""

import json
from pathlib import Path

import polars as pl

from ncaa_mbb_data_build.derived import rosters, schedule, team_ids

FIXTURES_DIR = (
    Path(__file__).parent / "tests" / "fixtures" / "raw_root" / "mbb" / "json"
)


def _load_finals() -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(FIXTURES_DIR.glob("*.json"))
    ]


def test_team_ids_2026_season():
    df = team_ids(2026)

    assert df.height > 300
    assert df.schema["id"] == pl.Utf8
    assert "team" in df.columns
    assert "season" in df.columns
    assert (df.get_column("season") == "2025-26").all()


def test_schedule_one_row_per_fixture_game():
    finals = _load_finals()
    df = schedule(finals, 2026)

    assert df.height == 8
    assert df.schema["contest_id"] == pl.Utf8
    assert "home" in df.columns
    assert "away" in df.columns
    assert "game_date" in df.columns
    assert (df.get_column("season") == 2026).all()

    expected_ids = {p.stem for p in FIXTURES_DIR.glob("*.json")}
    assert set(df.get_column("contest_id").to_list()) == expected_ids


def test_rosters_distinct_team_player_per_season():
    finals = _load_finals()
    df = rosters(finals, 2026)

    assert df.height > 0
    assert df.schema["team"] == pl.Utf8
    assert df.schema["player"] == pl.Utf8
    assert (df.get_column("season") == 2026).all()
    assert df.select("team", "player").is_duplicated().sum() == 0
