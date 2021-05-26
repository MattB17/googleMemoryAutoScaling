import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from MemoryAutoScaling.utils import append_aggregate_stats_for_time_series


AGG_STR = "MemoryAutoScaling.utils.aggregate_time_series"
RENAME_STR = "MemoryAutoScaling.utils.rename_aggregate_stats"


@pytest.fixture(scope="function")
def mock_ts():
    return np.array([1.3, 1.4, 1.5, 2.5, 2.7, 2.3, 3.7, 3.3, 3.5, 4.5])


@pytest.fixture(scope="function")
def mock_df(mock_ts):
    return pd.DataFrame({
        "max_mem": mock_ts.tolist(),
        "max_cpu": [1.3, 1.3, 2.1, 0.6, 0.5, 0.5, 0.7, 0.9, 1.1, 2.0]})


def test_with_empty_dataframe():
    df = pd.DataFrame({"mem": [], "cpu": []})
    empty_dict = {"avg": [], "max": [], "range": [], "median": [], "std": []}
    renamed_dict = {"mem_ts": [], "mem_max": [], "mem_range": [],
                    "mem_median": [], "mem_std": []}
    with patch(AGG_STR, return_value=empty_dict) as mock_agg, \
        patch(RENAME_STR, return_value=renamed_dict) as mock_rename:
        result = append_aggregate_stats_for_time_series(
            df, "mem", {}, 3, False)
    assert result == renamed_dict
    assert mock_agg.call_count == 1
    assert mock_agg.call_args[0][0].tolist() == []
    assert mock_agg.call_args[0][1] == 3
    mock_rename.assert_called_once_with(empty_dict, "mem", False)


def test_with_one_interval(mock_df, mock_ts):
        stats_dict = {"max": [4.50], "avg": [2.67], "median": [2.60],
                      "range": [3.20], "std": [1.03]}
        renamed_dict = {"max_mem_ts": [4.50], "max_mem_avg": [2.67],
                        "max_mem_median": [2.60], "max_mem_range": [3.20],
                        "max_mem_std": [1.03]}
        with patch(AGG_STR, return_value=stats_dict) as mock_agg, \
            patch(RENAME_STR, return_value=renamed_dict) as mock_rename:
            result = append_aggregate_stats_for_time_series(
                mock_df, "max_mem", {}, 10, True)
        assert result == renamed_dict
        assert mock_agg.call_count == 1
        assert mock_agg.call_args[0][0].tolist() == mock_ts.tolist()
        assert mock_agg.call_args[0][1] == 10
        mock_rename.assert_called_once_with(stats_dict, "max_mem", True)


def test_with_multi_interval(mock_df, mock_ts):
    stats_dict = {"max": [1.50, 2.70, 3.70],
                  "avg": [1.40, 2.50, 3.50],
                  "median": [1.40, 2.50, 3.50],
                  "range": [0.20, 0.40, 0.40],
                  "std": [0.08, 0.16, 0.16]}
    renamed_dict = {"max_mem_ts": [1.50, 2.70, 3.70],
                    "max_mem_avg": [1.40, 2.50, 3.50],
                    "max_mem_median": [1.40, 2.50, 3.50],
                    "max_mem_range": [0.20, 0.40, 0.40],
                    "max_mem_std": [0.08, 0.16, 0.16]}
    old_dict = {"max_cpu_ts": [1.35, 2.10, 2.45],
                "max_cpu_avg": [1.20, 2.10, 2.10],
                "max_cpu_median": [1.25, 2.01, 2.11],
                "max_cpu_range": [0.45, 0.0, 0.91],
                "max_cpu_std": [0.05, 0.00, 0.21]}
    expected_dict = {"max_cpu_ts": [1.35, 2.10, 2.45],
                     "max_cpu_avg": [1.20, 2.10, 2.10],
                     "max_cpu_median": [1.25, 2.01, 2.11],
                     "max_cpu_range": [0.45, 0.0, 0.91],
                     "max_cpu_std": [0.05, 0.00, 0.21],
                     "max_mem_ts": [1.50, 2.70, 3.70],
                     "max_mem_avg": [1.40, 2.50, 3.50],
                     "max_mem_median": [1.40, 2.50, 3.50],
                     "max_mem_range": [0.20, 0.40, 0.40],
                     "max_mem_std": [0.08, 0.16, 0.16]}
    with patch(AGG_STR, return_value=stats_dict) as mock_agg, \
        patch(RENAME_STR, return_value=renamed_dict) as mock_rename:
        result = append_aggregate_stats_for_time_series(
            mock_df, "max_mem", old_dict, 3, True)
    assert result == expected_dict
    assert mock_agg.call_count == 1
    assert mock_agg.call_args[0][0].tolist() == mock_ts.tolist()
    assert mock_agg.call_args[0][1] == 3
    mock_rename.assert_called_once_with(stats_dict, "max_mem", True)
