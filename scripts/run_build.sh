#!/usr/bin/env bash
# Build one or all NCAA MBB datasets for a season, offline against the
# sibling ncaa-mbb-hoops-raw checkout (or NCAA_MBB_RAW_ROOT if set).
#
# Usage:
#   SEASON=2026 bash scripts/run_build.sh
#   SEASON=2026 DATASET=shots bash scripts/run_build.sh
#
# Watch a running build live in another terminal:
#   tail -f scripts/../logs/run_build_<timestamp>.log
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Sibling raw checkout by default; an already-set NCAA_MBB_RAW_ROOT (or HTTP
# fallback URL) always wins over this relative default.
export NCAA_MBB_RAW_ROOT="${NCAA_MBB_RAW_ROOT:-$REPO_ROOT/../ncaa-mbb-hoops-raw}"
# Optional: point ingest's cache at a specific dir (unset = ingest default).
[ -n "${NCAA_MBB_CACHE:-}" ] && export NCAA_MBB_CACHE

mkdir -p logs
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGFILE="logs/run_build_${TIMESTAMP}.log"

uv run python -m ncaa_mbb_data_build build \
  --dataset "${DATASET:-all}" \
  --season "${SEASON:?set SEASON}" \
  2>&1 | tee "$LOGFILE"
