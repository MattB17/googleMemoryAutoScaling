import pytest
import numpy as np
from unittest.mock import patch
from MemoryAutoScaling.utils import aggregate_time_series


STATS_STR = "MemoryAutoScaling.utils.get_trace_stats"


@pytest.fixture(scope="function")
def mock_ts():
    return np.array([1.3, 1.4, 1.5, 2.5, 2.7, 2.3, 3.7, 3.3, 3.5, 4.5])


def test_with_no_data():
    with patch(STATS_STR) as mock_stats:
        result_dict = aggregate_time_series(np.array([]), 3)
    mock_stats.assert_not_called()
    assert result_dict['max'] == []
    assert result_dict['avg'] == []
    assert result_dict['median'] == []
    assert result_dict['range'] == []
    assert result_dict['std'] == []


def test_with_one_interval(mock_ts):
    stats_dict = {"max": 4.50, "avg": 2.67, "median": 2.60,
                  "range": 3.20, "std": 1.03}
    with patch(STATS_STR, return_value=stats_dict) as mock_stats:
        result_dict = aggregate_time_series(mock_ts, 10)
    assert mock_stats.call_count == 1
    assert mock_stats.call_args[0][0].tolist() == mock_ts.tolist()
    assert result_dict['max'] == [4.50]
    assert result_dict['avg'] == [2.67]
    assert result_dict['median'] == [2.60]
    assert result_dict['range'] == [3.20]
    assert result_dict['std'] == [1.03]


def test_with_multi_interval(mock_ts):
    stats_dict1 = {'max': 1.50, 'avg': 1.40, "median": 1.40,
                   'range': 0.20, 'std': 0.08}
    stats_dict2 = {'max': 2.70, 'avg': 2.50, 'median': 2.50,
                   'range': 0.40, 'std': 0.16}
    stats_dict3 = {'max': 3.70, 'avg': 3.50, 'median': 3.50,
                   'range': 0.40, 'std': 0.16}
    with patch(STATS_STR,
        side_effect=(stats_dict1, stats_dict2, stats_dict3)) as mock_stats:
        result_dict = aggregate_time_series(mock_ts, 3)
    assert mock_stats.call_count == 3
    assert result_dict['max'] == [1.50, 2.70, 3.70]
    assert result_dict['avg'] == [1.40, 2.50, 3.50]
    assert result_dict['median'] == [1.40, 2.50, 3.50]
    assert result_dict['range'] == [0.20, 0.40, 0.40]
    assert result_dict['std'] == [0.08, 0.16, 0.16]
