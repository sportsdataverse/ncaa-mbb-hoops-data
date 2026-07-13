"""Tests for build.py -- build_season driver, hermetic (real fixtures, no network)."""

import json
from pathlib import Path

import polars as pl

from ncaa_mbb_data_build.build import build_season

RAW_ROOT = Path(__file__).parent / "tests" / "fixtures" / "raw_root"
JSON_DIR = RAW_ROOT / "mbb" / "json"


def _fixture_family_height(family: str) -> int:
    total = 0
    for p in JSON_DIR.glob("*.json"):
        final = json.loads(p.read_text(encoding="utf-8"))
        total += len(final.get(family) or [])
    return total


def test_build_season_pbp_direct(tmp_path: Path):
    out = build_season("pbp", 2026, base=tmp_path, raw_root=RAW_ROOT)

    assert out.height == _fixture_family_height("pbp")
    assert out.select("contest_id").n_unique() == 8
    assert out.schema["contest_id"] == pl.Utf8
    assert (out.get_column("season") == 2026).all()

    pq = tmp_path / "mbb" / "pbp" / "parquet" / "pbp_2026.parquet"
    assert pq.exists()
    assert pl.read_parquet(pq).height == out.height


def test_build_season_shots_direct(tmp_path: Path):
    out = build_season("shots", 2026, base=tmp_path, raw_root=RAW_ROOT)

    assert out.height == _fixture_family_height("shots")

    pq = tmp_path / "mbb" / "shots" / "parquet" / "shots_2026.parquet"
    assert pq.exists()
    assert pl.read_parquet(pq).height == out.height


def test_build_season_schedule_derived(tmp_path: Path):
    out = build_season("schedule", 2026, base=tmp_path, raw_root=RAW_ROOT)

    assert out.height == 8
    pq = tmp_path / "mbb" / "schedule" / "parquet" / "schedule_2026.parquet"
    assert pq.exists()


def test_build_season_team_ids_derived(tmp_path: Path):
    out = build_season("team_ids", 2026, base=tmp_path, raw_root=RAW_ROOT)

    assert out.height > 300
    pq = tmp_path / "mbb" / "team_ids" / "parquet" / "team_ids_2026.parquet"
    assert pq.exists()


def test_build_season_no_contest_ids_returns_empty_and_writes_nothing(tmp_path: Path):
    out = build_season("pbp", 1999, base=tmp_path, raw_root=RAW_ROOT)

    assert out.height == 0
    assert not (tmp_path / "mbb" / "pbp").exists()
