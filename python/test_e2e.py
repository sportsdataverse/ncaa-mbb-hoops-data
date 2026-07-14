"""End-to-end offline build test -- all 9 datasets, one season, hermetic fixtures.

Builds every ``config.REGISTRY`` dataset from the 8 fixture games into a
single ``tmp_path`` base and asserts, per dataset: the parquet was written,
is non-empty, and the returned frame's schema round-trips through disk
unchanged. Also locks the dtype-discipline contract (``contest_id``/``id``
Utf8, ``season`` Int64 == 2026) and that a build upserts the manifest.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ncaa_mbb_data_build.build import build_season
from ncaa_mbb_data_build.config import REGISTRY

RAW_ROOT = Path(__file__).parent / "tests" / "fixtures" / "raw_root"

# The 6 DIRECT datasets + schedule all carry contest_id (Utf8, dtype discipline).
_HAS_CONTEST_ID = {
    "pbp",
    "lineups",
    "possessions",
    "player_box",
    "team_box",
    "shots",
    "schedule",
}
# The 6 DIRECT datasets carry a per-row season column pinned to the build season.
_DIRECT = {"pbp", "lineups", "possessions", "player_box", "team_box", "shots"}


def test_build_all_datasets(tmp_path: Path):
    for ds in REGISTRY:
        df = build_season(ds, 2026, base=tmp_path, raw_root=str(RAW_ROOT))

        pq = tmp_path / "mbb" / ds / "parquet" / f"{ds}_2026.parquet"
        assert pq.exists(), f"{ds}: parquet not written"

        on_disk = pl.read_parquet(pq)
        assert on_disk.height > 0, f"{ds}: parquet is empty"
        assert df.schema == on_disk.schema, f"{ds}: returned schema != on-disk schema"

        if ds in _HAS_CONTEST_ID:
            assert df.schema["contest_id"] == pl.Utf8, f"{ds}: contest_id not Utf8"
        if ds == "team_ids":
            assert df.schema["id"] == pl.Utf8, "team_ids: id not Utf8"
        if ds in _DIRECT:
            assert (df.get_column("season") == 2026).all(), f"{ds}: season != 2026"

    manifest = tmp_path / "mbb" / "pbp" / "manifest.csv"
    assert manifest.exists(), "pbp manifest.csv not written"
    rows = pl.read_csv(manifest)
    assert ((rows["dataset"] == "pbp") & (rows["season"] == 2026)).any(), (
        "pbp manifest missing a (pbp, 2026) row"
    )
