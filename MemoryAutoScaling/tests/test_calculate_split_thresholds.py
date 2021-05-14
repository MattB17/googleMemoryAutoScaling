import pytest
from unittest.mock import MagicMock, patch
from MemoryAutoScaling.utils import calculate_split_thresholds


THRESH_STR = "MemoryAutoScaling.utils.get_train_test_thresholds"


@pytest.fixture(scope="function")
def mock_data():
    return MagicMock()


def test_with_tuning(mock_data):
    with patch(THRESH_STR, return_value=(60, 20)) as mock_thresh:
        train_thresh, test_thresh = calculate_split_thresholds(
            mock_data, 0.603, 0.2015, True)
    mock_thresh.assert_called_once_with(mock_data, 0.60, 0.20)
    assert train_thresh == 60
    assert test_thresh == 20


def test_not_tuning(mock_data):
    with patch(THRESH_STR, return_value=(80, 20)) as mock_thresh:
        train_thresh, test_thresh = calculate_split_thresholds(
            mock_data, 0.6045, 0.2027, False)
    mock_thresh.assert_called_once_with(mock_data, 0.80, 0.20)
    assert train_thresh == 80
    assert test_thresh == 20
