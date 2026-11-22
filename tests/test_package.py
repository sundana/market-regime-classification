from __future__ import annotations

from importlib.metadata import entry_points, version

import market_regime_classification as package


def test_version_matches_installed_metadata():
    assert version("market-regime-classification") == package.__version__


def test_console_script_is_registered():
    scripts = entry_points(group="console_scripts")
    matching = [ep for ep in scripts if ep.name == "market-regime"]
    assert matching, "market-regime console script is not installed"
    assert matching[0].value == "market_regime_classification.run:main"
