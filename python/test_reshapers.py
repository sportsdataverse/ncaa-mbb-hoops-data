"""Tests for reshapers.py -- direct family extractor, hermetic (real fixture, no network)."""

import json
from pathlib import Path

import polars as pl

from ncaa_mbb_data_build.reshapers import extract_family

FIXTURE = (
    Path(__file__).parent
    / "tests"
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
