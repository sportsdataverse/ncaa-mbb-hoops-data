"""Tests for ingest.py -- LOCAL mode only, hermetic (no network)."""

import json

import polars as pl

from ncaa_mbb_data_build.ingest import read_parsed, season_contest_ids

PARSED = {
    "contest_id": "12345",
    "pbp": [{"a": 1}],
    "lineups": [],
    "player_box": [{"b": 2}],
    "team_box": [{"c": 3}],
    "shots": [],
    "possessions": [{"d": 4}],
}


def test_read_parsed_returns_exact_dict(tmp_path):
    json_dir = tmp_path / "mbb" / "json"
    json_dir.mkdir(parents=True)
    (json_dir / "12345.json").write_text(json.dumps(PARSED), encoding="utf-8")

    result = read_parsed("12345", raw_root=tmp_path)

    assert result == PARSED
    assert isinstance(result["contest_id"], str)


def test_read_parsed_missing_file_returns_none(tmp_path):
    assert read_parsed("00000", raw_root=tmp_path) is None


def test_read_parsed_malformed_json_returns_none(tmp_path):
    json_dir = tmp_path / "mbb" / "json"
    json_dir.mkdir(parents=True)
    (json_dir / "99999.json").write_text("{not valid json", encoding="utf-8")

    assert read_parsed("99999", raw_root=tmp_path) is None


def test_season_contest_ids_filters_and_sorts(tmp_path):
    (tmp_path / "mbb").mkdir()
    df = pl.DataFrame(
        {
            "contest_id": ["b", "a", "c"],
            "season": ["2026", "2026", "2025"],
            "captured": [True, True, True],
        },
        schema={"contest_id": pl.Utf8, "season": pl.Utf8, "captured": pl.Boolean},
    )
    df.write_parquet(tmp_path / "mbb" / "schedule_master.parquet")

    ids = season_contest_ids(2026, raw_root=tmp_path)

    assert ids == ["a", "b"]
    assert all(isinstance(i, str) for i in ids)


def test_season_contest_ids_missing_parquet_returns_empty(tmp_path):
    assert season_contest_ids(2026, raw_root=tmp_path) == []
