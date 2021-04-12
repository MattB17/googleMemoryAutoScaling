"""This module holds specification parameters for the Google Memory Auto
Scaling project. These are a set of parameters used throughout the project.

"""
from itertools import product


TRACE_ID_COL = "collection_id"
START_INTERVAL_COL = "start_time"
END_INTERVAL_COL = "end_time"
AVG_CPU_COL = "average_usage.cpus"
MAX_CPU_COL = "maximum_usage.cpus"
AVG_MEM_COL = "average_usage.memory"
MAX_MEM_COL = "maximum_usage.memory"
MAX_MEM_TS = "{}_ts".format(MAX_MEM_COL)
MAX_CPU_TS = "{}_ts".format(MAX_CPU_COL)
LAGS = [1, 2, 3]
MODELS_COUNT = 1
OVERALL_MASE_WEIGHT = 5
EPS = 0.00001

RAW_TIME_SERIES_COLS = [MAX_MEM_COL, AVG_MEM_COL, MAX_CPU_COL, AVG_CPU_COL]
RAW_TIME_SERIES_NAMES = ["Maximum Memory", "Average Memory",
                         "Maximum CPU", "Average CPU"]

MAX_STATS_COLS = ["ts", "avg", "std", "median", "range"]
AVG_STATS_COLS = ["ts", "max", "std", "median", "range"]

MODELING_COLS = ["params", "train_mase", "test_mase",
                 "under_mase", "prop_under_preds", "max_under_pred",
                 "over_mase", "prop_over_preds", "avg_over_pred"]


def get_trace_columns():
    """The columns in a trace dataframe.

    Returns
    -------
    list
        A list of strings representing the names of the columns in a
        trace dataframe.

    """
    max_cols = ["{0}_{1}".format(ts_name, stat_name) for ts_name, stat_name
                in product([MAX_MEM_COL, MAX_CPU_COL], MAX_STATS_COLS)]
    avg_cols = ["{0}_{1}".format(ts_name, stat_name) for ts_name, stat_name
                in product([AVG_MEM_COL, AVG_CPU_COL], AVG_STATS_COLS)]
    return max_cols + avg_cols


def get_lagged_trace_columns(lags, exclude_cols=None):
    """The column names for lagged data in a trace dataframe.

    Column names are generated for each lag in `lags`. Any columns specified
    in `exclude_cols` are excluded. If exclude_cols is None, no columns are
    excluded.

    Parameters
    ----------
    lags: list
        A list of integers representing the lags for the columns.
    exclude_cols: list
        A list of the columns to exclude. The default value is None.

    Returns
    -------
    list
        A list of strings representing the names of the lagged columns in a
        trace dataframe.

    """
    excluded_columns = exclude_cols if exclude_cols else []
    return ["{0}_lag_{1}".format(col_name, lag) for lag, col_name
            in product(lags, get_trace_columns())
            if col_name not in excluded_columns]
