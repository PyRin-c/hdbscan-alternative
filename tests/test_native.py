from importlib.metadata import version

import hdbscan_alternative


def test_native_version_matches_distribution():
    assert hdbscan_alternative.__version__ == version("hdbscan-alternative")
