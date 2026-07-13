"""Python producer for the ESPN NCAA MBB release datasets.

Cloned from ``wehoop-wnba-data/python/wnba_data_build`` (see that repo's
``espn_wnba_*_creation.R`` scripts for the R parity lineage). Reshapes the
sibling ``ncaa-mbb-hoops-raw`` per-game JSON into season-level parquet/csv +
manifest and publishes to the ``espn_mbb_*`` release tags.

NOTE: ``config.py``, ``ingest.py``, and ``reshapers.py`` are still the
wnba_data_build originals (WNBA-flavored env vars / tags / sportsdataverse.wnba
imports) copied verbatim as a scaffold baseline -- they are rewritten for NCAA
MBB in later tasks of this build.
"""

__all__ = ["config", "ingest", "io", "build", "publish", "reshapers"]
