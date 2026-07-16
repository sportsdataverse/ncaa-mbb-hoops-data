"""Tests for derived.py -- the 3 non-family datasets, hermetic (real fixtures, no network)."""

import json
from pathlib import Path

import polars as pl

from ncaa_mbb_data_build.derived import rosters, schedule, team_ids
from ncaa_mbb_data_build.reshapers import extract_family

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
    assert df.schema["season"] == pl.Int64
    assert (df.get_column("season") == 2026).all()


def test_team_ids_season_is_int64_and_joins_direct_dataset():
    """The real bug Fix 2 prevents: team_ids's season must actually JOIN
    against a DIRECT dataset's season (Int64 ending-year), not just carry
    the right dtype in isolation -- a Utf8 "2025-26" season here would
    silently zero-row every such join (pbp.join(team_ids, on="season"))."""
    finals = _load_finals()
    pbp = pl.concat(
        [
            extract_family(f, "pbp", season=2026, contest_id=f["contest_id"])
            for f in finals
        ],
        how="diagonal_relaxed",
    )
    ids = team_ids(2026)

    assert ids.schema["season"] == pl.Int64
    assert ids.get_column("season").unique().to_list() == [2026]

    joined = pbp.join(ids, on="season", how="inner")
    assert joined.height > 0


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


def test_rosters_empty_and_populated_share_schema():
    """The empty fallback and the populated path must agree on dtypes --
    notably games is Int64 (n_unique returns UInt32), so the season parquet
    schema stays stable whether or not a season has games."""
    populated = rosters(_load_finals(), 2026)
    empty = rosters([], 2026)
    assert empty.height == 0
    assert empty.schema == populated.schema
    assert populated.schema["games"] == pl.Int64
    assert empty.schema["games"] == pl.Int64


def test_schedule_final_score_is_max_not_opening_row():
    """Locks in .max() of pbp home/away score -- not pbp[0] (which is 0-0 opening tip)."""
    finals = _load_finals()
    df = schedule(finals, 2026).filter(pl.col("contest_id") == "1613299")

    assert df.height == 1
    # Verified against the fixture's pbp rows: game final, Maryland 78 - Illinois 67.
    assert df.get_column("home_score").item() == 78
    assert df.get_column("away_score").item() == 67


def test_schedule_and_rosters_skip_empty_family_without_raising():
    """A game with empty pbp/player_box must not abort the season build."""
    real = _load_finals()[0]
    finals = [real, {"contest_id": "ZZZ", "pbp": [], "player_box": []}]

    sched = schedule(finals, 2026)
    assert sched.height == 1
    assert "ZZZ" not in sched.get_column("contest_id").to_list()

    rost = rosters(finals, 2026)
    assert rost.height == rosters([real], 2026).height
