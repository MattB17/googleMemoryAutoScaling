import pytest
import pandas as pd
from unittest.mock import patch, call
from MemoryAutoScaling import specs
from MemoryAutoScaling.utils import build_trace_aggregate_df


APPEND_STR = "MemoryAutoScaling.utils.append_aggregate_stats_for_time_series"


@pytest.fixture(scope="function")
def mock_df():
    return pd.DataFrame(
        {specs.MAX_MEM_COL: [1.3, 1.4, 1.5, 2.5, 2.7, 2.3, 3.7, 3.3, 3.5, 4.5],
         specs.MAX_CPU_COL: [1.3, 1.3, 2.1, 0.6, 0.5, 0.5, 0.7, 0.9, 1.1, 2.0],
         specs.AVG_MEM_COL: [1.1, 1.1, 1.1, 1.9, 2.1, 2.3, 3.1, 2.9, 2.7, 4.0],
         specs.AVG_CPU_COL: [1.0, 1.0, 1.0, 0.3, 0.2, 0.2, 0.7, 0.6, 0.6, 1.0]})


def test_with_empty_dataframe():
    df = pd.DataFrame({specs.MAX_MEM_COL: [], specs.MAX_CPU_COL: [],
                       specs.AVG_MEM_COL: [], specs.AVG_CPU_COL: []})
    dict0 = {}
    dict1 = {'max_mem_ts': [], 'max_mem_avg': [], 'max_mem_range': [],
             'max_mem_std': [], 'max_mem_median': []}
    dict2 = {'max_mem_ts': [], 'max_mem_avg': [], 'max_mem_range': [],
             'max_mem_std': [], 'max_mem_median': [],
             'max_cpu_ts': [], 'max_cpu_avg': [], 'max_cpu_range': [],
             'max_cpu_std': [], 'max_cpu_median': []}
    dict3 = {'max_mem_ts': [], 'max_mem_avg': [], 'max_mem_range': [],
             'max_mem_std': [], 'max_mem_median': [],
             'max_cpu_ts': [], 'max_cpu_avg': [], 'max_cpu_range': [],
             'max_cpu_std': [], 'max_cpu_median': [],
             'avg_mem_max': [], 'avg_mem_ts': [], 'avg_mem_range': [],
             'avg_mem_std': [], 'avg_mem_median': []}
    dict4 = {'max_mem_ts': [], 'max_mem_avg': [], 'max_mem_range': [],
             'max_mem_std': [], 'max_mem_median': [],
             'max_cpu_ts': [], 'max_cpu_avg': [], 'max_cpu_range': [],
             'max_cpu_std': [], 'max_cpu_median': [],
             'avg_mem_max': [], 'avg_mem_ts': [], 'avg_mem_range': [],
             'avg_mem_std': [], 'avg_mem_median': [],
             'avg_cpu_max': [], 'avg_cpu_ts': [], 'avg_cpu_range': [],
             'avg_cpu_std': [], 'avg_cpu_median': []}
    return_dicts = (dict1, dict2, dict3, dict4)
    with patch(APPEND_STR, side_effect=return_dicts) as mock_append:
        result_df = build_trace_aggregate_df(df, 3)
    for ts_name in dict4.keys():
        assert result_df[ts_name].values.tolist() == dict4[ts_name]
    append_calls = [call(df, specs.MAX_MEM_COL, dict0, 3, True),
                    call(df, specs.MAX_CPU_COL, dict1, 3, True),
                    call(df, specs.AVG_MEM_COL, dict2, 3, False),
                    call(df, specs.AVG_CPU_COL, dict3, 3, False)]
    mock_append.assert_has_calls(append_calls)
    assert mock_append.call_count == 4


def test_with_one_interval(mock_df):
    dict0 = {}
    dict1 = {'max_mem_ts': [4.50], 'max_mem_avg': [2.67],
             'max_mem_range': [3.20], 'max_mem_std': [1.03],
             'max_mem_median': [2.60]}
    dict2 = {'max_mem_ts': [4.50], 'max_mem_avg': [2.67],
             'max_mem_range': [3.20], 'max_mem_std': [1.03],
             'max_mem_median': [2.60],
             'max_cpu_ts': [2.10], 'max_cpu_avg': [1.10],
             'max_cpu_range': [1.60], 'max_cpu_std': [0.55],
             'max_cpu_median': [1.0]}
    dict3 = {'max_mem_ts': [4.50], 'max_mem_avg': [2.67],
             'max_mem_range': [3.20], 'max_mem_std': [1.03],
             'max_mem_median': [2.60],
             'max_cpu_ts': [2.10], 'max_cpu_avg': [1.10],
             'max_cpu_range': [1.60], 'max_cpu_std': [0.55],
             'max_cpu_median': [1.0],
             'avg_mem_max': [4.00], 'avg_mem_ts': [2.23],
             'avg_mem_range': [2.90], 'avg_mem_std': [0.92],
             'avg_mem_median': [2.20]}
    dict4 = {'max_mem_ts': [4.50], 'max_mem_avg': [2.67],
             'max_mem_range': [3.20], 'max_mem_std': [1.03],
             'max_mem_median': [2.60],
             'max_cpu_ts': [2.10], 'max_cpu_avg': [1.10],
             'max_cpu_range': [1.60], 'max_cpu_std': [0.55],
             'max_cpu_median': [1.0],
             'avg_mem_max': [4.00], 'avg_mem_ts': [2.23],
             'avg_mem_range': [2.90], 'avg_mem_std': [0.92],
             'avg_mem_median': [2.20],
             'avg_cpu_max': [1.0], 'avg_cpu_ts': [0.66],
             'avg_cpu_range': [0.80], 'avg_cpu_std': [0.32],
             'avg_cpu_median': [0.65]}
    return_dicts = (dict1, dict2, dict3, dict4)
    with patch(APPEND_STR, side_effect=return_dicts) as mock_append:
        result_df = build_trace_aggregate_df(mock_df, 10)
    for ts_name in dict4.keys():
        assert result_df[ts_name].values.tolist() == dict4[ts_name]
    append_calls = [call(mock_df, specs.MAX_MEM_COL, dict0, 10, True),
                    call(mock_df, specs.MAX_CPU_COL, dict1, 10, True),
                    call(mock_df, specs.AVG_MEM_COL, dict2, 10, False),
                    call(mock_df, specs.AVG_CPU_COL, dict3, 10, False)]
    mock_append.assert_has_calls(append_calls)
    assert mock_append.call_count == 4


def test_with_multi_interval(mock_df):
    dict0 = {}
    dict1 = {"max_mem_ts": [1.50, 2.70, 3.70],
             "max_mem_avg": [1.40, 2.50, 3.50],
             "max_mem_median": [1.40, 2.50, 3.50],
             "max_mem_range": [0.20, 0.40, 0.40],
             "max_mem_std": [0.08, 0.16, 0.16]}
    dict2 = {"max_mem_ts": [1.50, 2.70, 3.70],
             "max_mem_avg": [1.40, 2.50, 3.50],
             "max_mem_median": [1.40, 2.50, 3.50],
             "max_mem_range": [0.20, 0.40, 0.40],
             "max_mem_std": [0.08, 0.16, 0.16],
             "max_cpu_ts": [1.35, 2.10, 2.45],
             "max_cpu_avg": [1.20, 2.10, 2.10],
             "max_cpu_median": [1.25, 2.01, 2.11],
             "max_cpu_range": [0.45, 0.0, 0.91],
             "max_cpu_std": [0.05, 0.00, 0.21]}
    dict3 = {"max_mem_ts": [1.50, 2.70, 3.70],
             "max_mem_avg": [1.40, 2.50, 3.50],
             "max_mem_median": [1.40, 2.50, 3.50],
             "max_mem_range": [0.20, 0.40, 0.40],
             "max_mem_std": [0.08, 0.16, 0.16],
             "max_cpu_ts": [1.35, 2.10, 2.45],
             "max_cpu_avg": [1.20, 2.10, 2.10],
             "max_cpu_median": [1.25, 2.01, 2.11],
             "max_cpu_range": [0.45, 0.0, 0.91],
             "max_cpu_std": [0.05, 0.00, 0.21],
             'avg_mem_max': [1.10, 2.30, 3.10],
             'avg_mem_ts': [1.10, 2.10, 2.90],
             'avg_mem_range': [0.00, 0.40, 0.40],
             'avg_mem_std': [0.00, 0.16, 0.16],
             'avg_mem_median': [1.10, 2.10, 2.90]}
    dict4 = {"max_mem_ts": [1.50, 2.70, 3.70],
             "max_mem_avg": [1.40, 2.50, 3.50],
             "max_mem_median": [1.40, 2.50, 3.50],
             "max_mem_range": [0.20, 0.40, 0.40],
             "max_mem_std": [0.08, 0.16, 0.16],
             "max_cpu_ts": [1.35, 2.10, 2.45],
             "max_cpu_avg": [1.20, 2.10, 2.10],
             "max_cpu_median": [1.25, 2.01, 2.11],
             "max_cpu_range": [0.45, 0.0, 0.91],
             "max_cpu_std": [0.05, 0.00, 0.21],
             'avg_mem_max': [1.10, 2.30, 3.10],
             'avg_mem_ts': [1.10, 2.10, 2.90],
             'avg_mem_range': [0.00, 0.40, 0.40],
             'avg_mem_std': [0.00, 0.16, 0.16],
             'avg_mem_median': [1.10, 2.10, 2.90],
             'avg_cpu_max': [1.00, 0.30, 0.70],
             'avg_cpu_ts': [1.00, 0.23, 0.63],
             'avg_cpu_range': [0.00, 0.10, 0.10],
             'avg_cpu_std': [0.00, 0.05, 0.05],
             'avg_cpu_median': [1.00, 0.20, 0.60]}
    return_dicts = (dict1, dict2, dict3, dict4)
    with patch(APPEND_STR, side_effect=return_dicts) as mock_append:
        result_df = build_trace_aggregate_df(mock_df, 3)
    for ts_name in dict4.keys():
        assert result_df[ts_name].values.tolist() == dict4[ts_name]
    append_calls = [call(mock_df, specs.MAX_MEM_COL, dict0, 3, True),
                    call(mock_df, specs.MAX_CPU_COL, dict1, 3, True),
                    call(mock_df, specs.AVG_MEM_COL, dict2, 3, False),
                    call(mock_df, specs.AVG_CPU_COL, dict3, 3, False)]
    mock_append.assert_has_calls(append_calls)
    assert mock_append.call_count == 4
