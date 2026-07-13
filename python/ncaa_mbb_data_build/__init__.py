"""Python producer for the NCAA MBB release datasets.

Cloned from ``wehoop-wnba-data/python/wnba_data_build`` as a scaffold baseline,
then rewritten for the NCAA (stats.ncaa.org / bigballR) lineage. Reshapes the
sibling ``ncaa-mbb-hoops-raw`` per-game JSON into season-level parquet/csv +
manifest and publishes to the ``ncaa_mbb_*`` release tags.
"""

__all__ = ["config", "ingest", "io", "build", "publish", "reshapers"]
