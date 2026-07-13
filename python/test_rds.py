"""Tests for rds.py -- Rscript arrow::read_parquet -> saveRDS conversion.

Gated: only runs when the resolved ``Rscript`` has R's ``arrow`` package.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import polars as pl
import pytest

from ncaa_mbb_data_build.rds import _find_rscript, to_rds

_rscript = _find_rscript()


def _has_arrow(rscript: str | None) -> bool:
    if rscript is None:
        return False
    result = subprocess.run(
        [rscript, "-e", 'cat(requireNamespace("arrow", quietly=TRUE))'],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "TRUE"


_has_rscript_with_arrow = _has_arrow(_rscript)

pytestmark = pytest.mark.skipif(
    not _has_rscript_with_arrow, reason="Rscript with arrow not available"
)


def test_to_rds_round_trips_through_r(tmp_path: Path):
    df = pl.DataFrame(
        {
            "game_id": [1, 2, 3],
            "team": ["Duke", "UNC", "Kansas"],
            "score": [72.5, 68.0, 81.25],
        }
    )
    parquet = tmp_path / "in.parquet"
    df.write_parquet(parquet)

    out = to_rds(parquet, tmp_path / "out.rds")

    assert out == tmp_path / "out.rds"
    assert out.exists()
    assert out.stat().st_size > 0

    result = subprocess.run(
        [
            _rscript,
            "-e",
            "x<-readRDS(commandArgs(TRUE)[1]); cat(nrow(x), ncol(x))",
            out.as_posix(),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    nrow, ncol = (int(x) for x in result.stdout.split())
    assert (nrow, ncol) == (df.height, df.width)
