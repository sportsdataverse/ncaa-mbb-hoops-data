"""Tests for publish.py -- per-file ``gh release upload --clobber``.

Hermetic: no real ``gh`` or R calls. Files are pre-staged; ``runner``/
``exists_check`` are fakes that just record/answer.
"""

from __future__ import annotations

from pathlib import Path

from ncaa_mbb_data_build.config import REGISTRY
from ncaa_mbb_data_build.publish import DEFAULT_REPO, publish_dataset

_SPEC = REGISTRY["pbp"]


def _stage(tmp_path: Path) -> None:
    pq_dir = tmp_path / "mbb" / "pbp" / "parquet"
    pq_dir.mkdir(parents=True)
    (pq_dir / "pbp_2026.parquet").write_bytes(b"parquet-bytes")

    rel_dir = tmp_path / "mbb" / "_release_build" / "pbp"
    rel_dir.mkdir(parents=True)
    (rel_dir / "pbp_2026.csv").write_bytes(b"csv-bytes")
    (rel_dir / "pbp_2026.rds").write_bytes(b"rds-bytes")


def test_publish_creates_release_when_absent(tmp_path: Path):
    _stage(tmp_path)
    calls: list[list[str]] = []

    result = publish_dataset(
        _SPEC,
        2026,
        base=tmp_path,
        runner=lambda args: calls.append(args),
        exists_check=lambda t, r: False,
        make_rds=False,
    )

    creates = [c for c in calls if c[:2] == ["release", "create"]]
    uploads = [c for c in calls if c[:2] == ["release", "upload"]]
    assert len(creates) == 1
    assert creates[0][2] == "ncaa_mbb_pbp"
    assert "--repo" in creates[0] and DEFAULT_REPO in creates[0]

    assert len(uploads) == 3
    for c in uploads:
        assert c[2] == "ncaa_mbb_pbp"
        assert c[4:6] == ["--repo", DEFAULT_REPO]
        assert c[-1] == "--clobber"

    assert result["uploaded"] == 3
    assert result["tag"] == "ncaa_mbb_pbp"


def test_publish_skips_create_when_release_present(tmp_path: Path):
    _stage(tmp_path)
    calls: list[list[str]] = []

    publish_dataset(
        _SPEC,
        2026,
        base=tmp_path,
        runner=lambda args: calls.append(args),
        exists_check=lambda t, r: True,
        make_rds=False,
    )

    assert not any(c[:2] == ["release", "create"] for c in calls)
    assert sum(1 for c in calls if c[:2] == ["release", "upload"]) == 3


def test_publish_dry_run_makes_no_calls(tmp_path: Path):
    _stage(tmp_path)
    calls: list[list[str]] = []

    result = publish_dataset(
        _SPEC,
        2026,
        base=tmp_path,
        dry_run=True,
        runner=lambda args: calls.append(args),
        exists_check=lambda t, r: False,
        make_rds=False,
    )

    assert calls == []
    assert result["uploaded"] == 0


def test_publish_only_parquet_staged_uploads_one_file(tmp_path: Path):
    pq_dir = tmp_path / "mbb" / "pbp" / "parquet"
    pq_dir.mkdir(parents=True)
    (pq_dir / "pbp_2026.parquet").write_bytes(b"parquet-bytes")
    calls: list[list[str]] = []

    result = publish_dataset(
        _SPEC,
        2026,
        base=tmp_path,
        runner=lambda args: calls.append(args),
        exists_check=lambda t, r: True,
        make_rds=False,
    )

    uploads = [c for c in calls if c[:2] == ["release", "upload"]]
    assert len(uploads) == 1
    assert result["uploaded"] == 1
