"""Tests for cli.py -- build subcommand, hermetic (real fixtures, no network, no gh)."""

from pathlib import Path

import polars as pl
import pytest

from ncaa_mbb_data_build.cli import main
from ncaa_mbb_data_build.config import REGISTRY

RAW_ROOT = str(Path(__file__).parent / "tests" / "fixtures" / "raw_root")


def test_build_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["build", "--help"])
    assert exc.value.code == 0


def test_build_dataset_shots(tmp_path: Path):
    rc = main(
        [
            "build",
            "--dataset",
            "shots",
            "--season",
            "2026",
            "--base",
            str(tmp_path),
            "--raw-root",
            RAW_ROOT,
        ]
    )
    assert rc == 0
    pq = tmp_path / "mbb" / "shots" / "parquet" / "shots_2026.parquet"
    assert pq.exists()
    assert pl.read_parquet(pq).height > 0


def test_build_dataset_all(tmp_path: Path):
    rc = main(
        [
            "build",
            "--dataset",
            "all",
            "--season",
            "2026",
            "--base",
            str(tmp_path),
            "--raw-root",
            RAW_ROOT,
        ]
    )
    assert rc == 0
    for dataset in REGISTRY:
        pq = tmp_path / "mbb" / dataset / "parquet" / f"{dataset}_2026.parquet"
        assert pq.exists(), f"missing parquet for {dataset}"
