import pytest
import numpy as np
from MemoryAutoScaling.utils import get_train_test_thresholds


@pytest.fixture(scope="function")
def mock_trace():
    return np.array([0.3, 0.4, 0.5, 0.9, 1.1, 1.7, 0.5, 0.3, 0.6, 1.1])


def test_train_and_val(mock_trace):
    train_thresh, test_thresh = get_train_test_thresholds(
        mock_trace, 0.6, 0.2)
    assert train_thresh == 6
    assert test_thresh == 8


def test_train_and_test(mock_trace):
    train_thresh, test_thresh = get_train_test_thresholds(
        mock_trace, 0.8, 0.2)
    assert train_thresh == 8
    assert test_thresh == 10


def test_capping_test(mock_trace):
    train_thresh, test_thresh = get_train_test_thresholds(
        mock_trace, 0.8, 0.4)
    assert train_thresh == 8
    assert test_thresh == 10


def test_rounding_thresholds():
    trace_ts = np.array([0.35, 0.43, 1.03, 1.71, 0.85, 0.68, 0.73,
                         1.37, 0.68, 1.15, 0.87, 0.63, 0.58, 1.11])
    train_thresh, test_thresh = get_train_test_thresholds(trace_ts, 0.7, 0.15)
    assert train_thresh == 10
    assert test_thresh == 12
