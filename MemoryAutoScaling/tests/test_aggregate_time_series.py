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
