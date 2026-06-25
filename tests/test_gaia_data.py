import pytest

import gaia_data


def test_gaia_data_download_disabled():
    with pytest.raises(RuntimeError, match="disabled"):
        gaia_data.get_gaia_data_dir()


def test_ground_truth_returns_none():
    assert gaia_data.get_ground_truth("any-id") is None
