"""This module holds specification parameters for the Google Memory Auto
Scaling project. These are a set of parameters used throughout the project.

"""
from itertools import product


TRACE_ID_COL = "collection_id"
START_INTERVAL_COL = "start_time"
END_INTERVAL_COL = "end_time"
TOTAL_MEM_COL = "assigned_memory"
AVG_CPU_COL = "average_usage.cpus"
MAX_CPU_COL = "maximum_usage.cpus"
AVG_MEM_COL = "average_usage.memory"
MAX_MEM_COL = "maximum_usage.memory"
MAX_MEM_TS = "{}_ts".format(MAX_MEM_COL)
MAX_CPU_TS = "{}_ts".format(MAX_CPU_COL)
LAGS = [1, 2, 3]
MODELS_COUNT = 1
HARVEST_WEIGHT = 3
EPS = 0.00001
CPU_ALLOC_PERCENTILE = 99.9

MULTI_VAR_COLS = [MAX_MEM_TS, MAX_CPU_TS]

RAW_TIME_SERIES_COLS = [MAX_MEM_COL, AVG_MEM_COL, MAX_CPU_COL, AVG_CPU_COL]
RAW_TIME_SERIES_NAMES = ["Maximum Memory", "Average Memory",
                         "Maximum CPU", "Average CPU"]

MAX_STATS_COLS = ["ts", "avg", "std", "median", "range"]
AVG_STATS_COLS = ["ts", "max", "std", "median", "range"]

MODELING_COLS = ["params", "train_mase", "test_mase", "under_mase",
                 "prop_under_preds", "max_under_pred", "over_mase",
                 "prop_over_preds", "avg_over_pred", "total_spare"]
BUFFER_PCTS = [0.0, 0.25, 0.5, 0.75, 1.0]
TUNING_BUFFER_PCT = BUFFER_PCTS[1]
RESULTS_COLS = [col for col in MODELING_COLS if col != "params"]
VIOLATIONS_WEIGHT = 3


def get_target_variable(max_mem):
    """Returns the target variable based on `max_mem`.

    If `max_mem` is True, Maximum Memory Usage is the target variable.
    Otherwise, it is Maximum CPU Usage.

    Parameters
    ----------
    max_mem: bool
        A boolean value indicating whether maximum memory usage is the target
        variable.

    Returns
    -------
    str
        A string representing the name of the target variable.

    """
    return MAX_MEM_TS if max_mem else MAX_CPU_TS


def get_input_and_output_target(max_mem):
    """Returns the input and output target variables based on `max_mem`.

    The input target variable is the name of the target variable in the raw
    trace data. The output target variable is the name of the target variable
    after pre-processing and is used in the modeling procedure. If `max_mem`
    is True, Maximum Memory Usage is used as the target variable. Otherwise,
    the target variable is Maximum CPU Usage.

    Parameters
    ----------
    max_mem: bool
        A boolean value indicating whether maximum memory usage is the target
        variable.

    Returns
    -------
    str, str
        Two strings representing the input and output target variables,
        respectively.

    """
    if max_mem:
        return MAX_MEM_COL, MAX_MEM_TS
    return MAX_CPU_COL, MAX_CPU_TS


def get_causal_cols(target_variable):
    """Gets the columns used to calculate causation against `target_variable`.

    Parameters
    ----------
    target_variable: str
        The name of the target variable.

    Returns
    -------
    list
        A list of strings representing the columns for which causation
        p-values will be calculated against `target_variable`.

    """
    raw_causal_cols = [col_name for col_name in RAW_TIME_SERIES_COLS
                       if col_name != target_variable]
    return ["{}_ts".format(col_name) for col_name in raw_causal_cols]


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
