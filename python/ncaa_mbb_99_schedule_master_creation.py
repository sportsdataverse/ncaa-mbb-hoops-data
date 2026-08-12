"""Stage 99 -- schedule master, games-in-data-repo manifest, and coverage index.

Runs LAST, after every dataset for every season is built: it reads what the
other stages committed. Thin shim over ``ncaa_mbb_data_build.master``; emits
all three artifacts from one in-memory frame so they cannot drift.

* ``mbb/ncaa_mbb_schedule_master.parquet`` -- every contest stats.ncaa.org
  lists (the denominator, from the RAW repo's D33 schedule master).
* ``mbb/ncaa_mbb_games_in_data_repo.parquet`` -- only contests with >=1
  ``in_*`` flag true (the numerator; what consumers join against).
* ``mbb/ncaa_mbb_schedule_coverage.parquet`` -- one row per season with
  per-dataset build coverage.

The ``in_*`` flag set is derived from ``config.REGISTRY`` (``level == "game"``)
and stamped from the COMMITTED per-season parquets -- exact, not a proxy.
Every flag is materialized, so a family with no coverage at all reports honest
zeros instead of vanishing from the schema.

Stage 99 is not a dataset shim: it has no registry entry and no ``DATASET``
constant. Number 99 is reserved for the schedule master (spec D34), which is
why ``tests/test_stage_inventory.py`` skips it.

Example:
    Rebuild the master from the committed tree + the sibling raw checkout::

        NCAA_MBB_RAW_ROOT=../ncaa-mbb-hoops-raw \
            uv run python python/ncaa_mbb_99_schedule_master_creation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ncaa_mbb_data_build.master import build_coverage, build_master, games_in_data_repo

REPO_ROOT = Path(__file__).resolve().parents[1]
LEAGUE = "mbb"
PREFIX = "ncaa_mbb"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base", default=str(REPO_ROOT), help="repo root holding the mbb/ tree"
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help="ncaa-mbb-hoops-raw checkout root (default: $NCAA_MBB_RAW_ROOT)",
    )
    args = parser.parse_args(argv)

    master = build_master(base=args.base, raw_root=args.raw_root)
    manifest = games_in_data_repo(master)
    coverage = build_coverage(master)

    out = Path(args.base) / LEAGUE
    out.mkdir(parents=True, exist_ok=True)
    for frame, name in (
        (master, "schedule_master"),
        (manifest, "games_in_data_repo"),
        (coverage, "schedule_coverage"),
    ):
        frame.write_parquet(out / f"{PREFIX}_{name}.parquet")

    seasons = master.get_column("season").unique().sort()
    print(f"master:   {master.height} contests, seasons {seasons[0]}-{seasons[-1]}")
    print(f"manifest: {manifest.height} contests in >=1 dataset")
    print(f"coverage: {coverage.height} seasons")
    for flag in sorted(c for c in master.columns if c.startswith("in_")):
        print(f"  {flag}: {master[flag].sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
