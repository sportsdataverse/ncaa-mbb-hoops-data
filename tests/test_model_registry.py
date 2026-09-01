"""models/REGISTRY.md carries one row per published RAPM estimand and names
its gates. Pure-file parser (ops/ scripts are not importable packages);
bites per-row: delete a tag's row and this fails.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "models" / "REGISTRY.md"

TAGS = ["ncaa_mbb_rapm", "ncaa_mbb_rapm_within_team"]
#: gate tokens that must appear in the league-wide row (names, not values —
#: values are frozen in ops/build_rapm_league.py's docstring, the authority).
LEAGUE_GATE_TOKENS = ["usable-possession", "intercept era band", "Torvik", "0.93"]


def _rows() -> list[str]:
    text = REGISTRY.read_text(encoding="utf-8")
    return [ln for ln in text.splitlines() if ln.startswith("|") and "---" not in ln]


def test_registry_exists():
    assert REGISTRY.is_file(), "models/REGISTRY.md is missing"


def test_each_estimand_has_a_row():
    rows = _rows()
    for tag in TAGS:
        assert any(f"`{tag}`" in r for r in rows), f"no registry row for {tag}"


def test_league_row_names_its_gates():
    row = next(r for r in _rows() if "`ncaa_mbb_rapm`" in r)
    missing = [t for t in LEAGUE_GATE_TOKENS if t not in row]
    assert not missing, f"league-wide row missing gate tokens: {missing}"


def test_estimands_not_conflated():
    """The two tags are different estimands; both rows must say 'estimand'."""
    rows = [r for r in _rows() if "rapm" in r]
    assert len(rows) >= 2
    assert all("estimand" in r for r in rows)
