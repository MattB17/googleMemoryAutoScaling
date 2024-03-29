"""A set of utility functions used throughout the memory auto scaling code.

"""
import copy
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
from MemoryAutoScaling import specs


def cap_predictions_at_1(predicted_vals):
    """Caps each value of `predicted_vals` at 1.

    That is, if any value of `predicted_vals` is greater than 1 then
    it is reset to 1. Otherwise, it is left unchanged.

    Parameters
    ----------
    predicted_vals: np.array
        A numpy array containing the predicted values to be capped.

    Returns
    -------
    np.array
        The numpy array obtained from `predicted_vals` after all of the
        predictions have been capped at 1.

    """
    return np.minimum(predicted_vals, 1.0)

def cap_and_clean_values(data_df, data_col, cap_val):
    """Cleans and caps the values of `data_col` in `data_df` using `cap_val`.

    Parameters
    ----------
    data_df: pd.DataFrame
        The pandas DataFrame containing the data.
    data_col: str
        A string representing the name of the column containing the data.
    cap_val: float
        The value used to cap the values of `data_col`.

    Returns
    -------
    np.array
        A numpy array representing the values of `data_col` from `data_df`
        after they have been cleaned and capped at `cap_val`.

    """
    data_vals = data_df[data_col].replace(np.nan, 0).values
    return np.minimum(data_vals, cap_val)


def cap_train_and_test_predictions(train_preds, test_preds):
    """Caps the training and testing predictions.

    Caps both the values of `train_preds` and `test_preds` at 1.

    Parameters
    ----------
    train_preds: np.array
        A numpy array representing the predictions for the train set.
    test_preds: np.array
        A numpy array representing the predictions for the test set.

    Returns
    -------
    np.array, np.array
        Two numpy arrays representing the predictions for the training and
        testing sets, repectively, both capped at 1.

    """
    return cap_predictions_at_1(train_preds), cap_predictions_at_1(test_preds)


def get_train_test_thresholds(data_trace, train_prop, test_prop):
    """Calculates the index identifying the end of the train and test sets.

    That is, an index of `data_trace` is calculated based on `train_prop`
    which identifies the training dataset of `data_trace`. And an index of
    `data_trace` is calculated based on `test_prop` which identifies the
    testing dataset of `data_trace`.

    Paraneters
    ----------
    data_trace: np.array
        A numpy array representing the data trace.
    train_prop: float
        A float in the range [0, 1] representing the proportion of data
        in the training set.
    test_prop: float
        A float in the range [0, 1] representing the proportion of data
        in the testing set.

    Returns
    -------
    int, int
        Two integers representing the indices of `data_trace` that identify
        the end of the training and testing sets, respectively.

    """
    n = len(data_trace)
    train_thresh = int(np.ceil(n * train_prop))
    test_end_pct = train_prop + test_prop
    test_thresh = n if test_end_pct >= 1.0 else int(np.ceil(n * test_end_pct))
    return train_thresh, test_thresh


def calculate_split_thresholds(data, train_prop, val_prop, tuning=True):
    """Calculates the split thresholds for `data`.

    If `tuning` is True, the split thresholds for `data` mark the end of
    the training and validation sets, respectively. Otherwise, the split
    thresholds mark the end of the training + validation and testing sets,
    respectively.

    Parameters
    ----------
    data: pd.DataFrame
        A pandas DataFrame representing the data to be split.
    train_prop: float
        A float representing the proportion of the training set.
    val_prop: float
        A float representing the proportion of the testing set.
    tuning: bool
        A boolean value indicating whether the split is for tuning or
        testing.

    Returns
    -------
    int, int
        Two thresholds representing the endpoint of the two subsets. If
        `tuning` is true these thresholds represent the end of the training
        and validation sets, respectively. Otherwise, they represent the end
        of the training + validation and testing sets, respectively.

    """
    train_prop = round(train_prop, 2)
    val_prop = round(val_prop, 2)
    if tuning:
        return get_train_test_thresholds(data, train_prop, val_prop)
    return get_train_test_thresholds(
        data, train_prop + val_prop, 1.0 - train_prop - val_prop)


def aggregate_time_series(time_series, window_length):
    """Max aggregates `time_series` at the period `window_length`.

    The aggregation computes the maximum, standard deviation, average, median,
    and range of `time_series` for each interval of length `window_length`.

    Parameters
    ----------
    time_series: np.array
        A numpy array representing the time series to be aggregated.
    window_length: int
        An integer denoting the aggregation interval. That is, every
        `window_length` time periods of `time_series` are aggregated.

    Returns
    -------
    dict
        A dictionary with the aggregate statistics for `time_series`. Each
        key is a string representing the statistic name and the corresponding
        value is a list representing the aggregated values of the statistic.

    """
    agg_dict = {"avg": [],
                "std": [],
                "median": [],
                "max": [],
                "range": []}
    intervals = len(time_series) // window_length
    for idx in range(intervals):
        start = window_length * idx
        end = window_length * (idx + 1)
        agg_stats = get_trace_stats(time_series[start:end])
        for stat in agg_stats.keys():
            agg_dict[stat].append(agg_stats[stat])
    return agg_dict


def rename_aggregate_stats(agg_stats, ts_name, is_max=True):
    """Renames the statistics in `agg_stats` based on `ts_name`.

    Parameters
    ----------
    agg_stats: dict
        A dictionary containing the aggregate stats to be renamed.
    ts_name: str
        A string representing the name of the time series.
    is_max: bool
        A boolean value indicating whether the time series records the
        maximum or not. The default value is True.

    Returns
    -------
    dict
        The dictionary obtained from `agg_stats` after renaming.

    """
    agg_stats = {"{0}_{1}".format(ts_name, stat_name): stat
                 for stat_name, stat in agg_stats.items()}
    ts_col = "{}_max".format(ts_name) if is_max else "{}_avg".format(ts_name)
    agg_stats["{}_ts".format(ts_name)] = agg_stats[ts_col]
    del agg_stats[ts_col]
    return agg_stats


def append_aggregate_stats_for_time_series(trace_df, ts_name, append_dict,
                                           window_length, is_max=True):
    """Appends aggregate stats for `ts_name` in `trace_df` to `append_dict`.

    The aggregate statistics for `ts_name` are calculated based on
    `window_length` and combined with the data in `append_dict`.

    Parameters
    ----------
    trace_df: pd.DataFrame
        A pandas DataFrame containing the trace data.
    ts_name: str
        A string representing the name of the time series.
    append_dict: dict
        The dictionary to which the aggregate stats are appended.
    window_length: int
        An integer representing the aggregation window.
    is_max: bool
        A boolean value indicating whether the time series records the
        maximum or not. The default value is True.

    Returns
    -------
    dict
        The dictionary obtained from `append_dict` after adding the
        aggregate statistics for `ts_name`.

    """
    ts = trace_df[ts_name].values
    agg_stats = rename_aggregate_stats(
        aggregate_time_series(ts, window_length), ts_name, is_max)
    return {**append_dict, **agg_stats}


def build_trace_aggregate_df(raw_trace_df, window_len):
    """Builds an aggregated dataframe from `raw_trace_df` using `window_len`.

    Parameters
    ----------
    raw_trace_df: pd.DataFrame
        A pandas DataFrame containing the raw data for the trace.
    window_len: int
        The aggregation window for the trace. Data will be aggregated every
        `window_len` time periods.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the aggregated measures for the trace
        data specified in `raw_trace_df` using `window_len` as the aggregation
        period.

    """
    trace_dict = {}
    for ts_name in [specs.MAX_MEM_COL, specs.MAX_CPU_COL]:
        trace_dict = append_aggregate_stats_for_time_series(
            raw_trace_df, ts_name, trace_dict, window_len, True)
    for ts_name in [specs.AVG_MEM_COL, specs.AVG_CPU_COL]:
        trace_dict = append_aggregate_stats_for_time_series(
            raw_trace_df, ts_name, trace_dict, window_len, False)
    return pd.DataFrame(trace_dict)


def get_trace_stats(data_trace):
    """Calculates high-level summary statistics for `data_trace`.

    Parameters
    ----------
    data_trace: np.array
        A numpy array representing the data trace for which the statistics
        are calculated.

    Returns
    -------
    dict
        A dictionary containing high-level summary statistics for `data_trace`.
        The keys are strings representing the name of the statistic and the
        values are the corresponding statistical value for `data_trace`.

    """
    if len(data_trace) > 0:
        max_val = np.max(data_trace)
        return {"avg": np.mean(data_trace),
                "std": np.std(data_trace),
                "median": np.median(data_trace),
                "max": max_val,
                "range": max_val - np.min(data_trace)}
    return {}


def perform_coin_toss(prob):
    """Performs a coin toss with probability `prob` of heads.

    Parameters
    ----------
    prob: float
        A number between 0 and 1 representing the probability of heads.

    Returns
    -------
    int
        Either 0 or 1 where 0 represents tails and 1 represents heads.

    """
    return np.random.binomial(1, prob)


def get_cumulative_sum_of_trace(data_trace):
    """Computes the cumulative sum of `data_trace`.

    That is, a new data trace is generated that is the same length as
    `data_trace` and for which at each time point the observation of the
    trace is equal to the cumulative sum of `data_trace` up to that time
    point.

    Parameters
    ----------
    data_trace: np.array
        A numpy array representing the data trace from which the cumulative
        sum is computed.

    Returns
    -------
    np.array
        A numpy array having the same length as `data_trace` and for which
        the observation at each time point is the cumulative sum of the
        observations of `data_trace` up to that time point.

    """
    avg = np.mean(data_trace)
    cumsum_trace = data_trace - avg
    cumsum_trace[0] = 0
    return np.cumsum(cumsum_trace) + avg


def impute_for_time_series(time_series, impute_val):
    """Performs imputation on `time_series` using `impute_val`.

    All NaN values in `time_series` are replaced by `impute_val`.

    Parameters
    ----------
    time_series: np.array
        A numpy array representing the time series for which imputation
        is performed.
    impute_val: float
        The value used for imputation.

    Returns
    -------
    np.array
        The numpy array obtained from `time_series` after replacing each NaN
        value with `impute_val`.

    """
    time_series[np.isnan(time_series)] = impute_val
    return time_series


def clean_resource_time_series(raw_trace_df, resource_col):
    """Cleans `resource_col` in `raw_trace_df`.

    The values of `resource_col` in `raw_trace_df` corresponding to NaN or
    infinity are set to 0.

    Parameters
    ----------
    raw_trace_df: pd.DataFrame
        A pandas DataFrame containing the data for the trace.
    resource_col: str
        A string representing the name of the resource for which the time
        series is cleaned.

    Returns
    -------
    np.array
        A numpy array corresponding to the values of `resource_col` in
        `raw_trace_df` after it has been cleaned.

    """
    resource_ts = raw_trace_df[resource_col].values
    resource_ts[np.isnan(resource_ts)] = 0
    resource_ts[np.isinf(resource_ts)] = 0
    return resource_ts


def calculate_max_allocated(raw_trace_df, resource_col):
    """Calculatate the maximum allocated amount over the life of the trace.

    Parameters
    ----------
    raw_trace_df: pd.DataFrame
        A pandas DataFrame containing the data for the trace.
    resource_col: str
        A string representing the name of the resource for which the maximum
        allocated amount is being calculated.

    Returns
    -------
    float
        A float representing the maximum allocated amount of `resource_col`
        over the life of the trace.

    """
    alloc_ts = clean_resource_time_series(raw_trace_df, resource_col)
    return np.max(alloc_ts)

def calculate_allocated_from_percentile(raw_trace_df, resource_col, p):
    """Calculates the allocated amount of `resource_col` based on `p`.

    The amount of `resource_col` allocated for the trace is calculated as
    the `p`th percentile of the values `resource_col` in `raw_trace_df`.

    Parameters
    ----------
    raw_trace_df: pd.DataFrame
        A pandas DataFrame containing the data for the trace.
    resource_col: str
        A string representing the name of the resource for which the `p`th
        percentile is calculated.
    p: float
        A float representing the percentile used to determine the amount
        allocated.

    Returns
    -------
    float
        A float representing the `p`th percentile value of `resource_col`.

    """
    alloc_ts = clean_resource_time_series(raw_trace_df, resource_col)
    return np.percentile(alloc_ts, p)


def build_trace_data_from_trace_df(raw_trace_df, agg_window, trace_usage):
    """Builds data for a `Trace` object from `raw_trace_df`.

    The DataFrame containing the data for a `Trace` is built. This is done
    by first replacing NaN values with 0 and then computing the aggregated
    statistics for `raw_trace_df` based on `agg_window`. The aggregated
    quantities are reported as percentages based on the raw usage data from
    `trace_usage`

    Parameters
    ----------
    trace_df: pd.DataFrame
        A pandas DataFrame containing the raw data for the trace.
    agg_window: int
        An integer representing the aggregation period.
    trace_usage: TraceUsage
        A `TraceUsage` object containing the usage data for the trace.

    Returns
    -------
    pd.DataFrame
        The DataFrame obtained from `raw_trace_df` after removing NaN values
        and aggregating data according to `agg_window`.

    """
    trace_df = compute_usage_percents_for_trace(raw_trace_df, trace_usage)
    trace_df = trace_df[specs.RAW_TIME_SERIES_COLS].fillna(0)
    return build_trace_aggregate_df(trace_df, agg_window)


def compute_usage_percents_for_trace(raw_trace_df, trace_usage):
    """Computes the usage percentages for `raw_trace_df`.

    The usage percentages are calculated by dividing the resource usage
    columns by the assigned resource amount. The resources of interest are
    memory and CPU. The usage statistics consist of average and maximum usage
    of each resource.

    Parameters
    ----------
    raw_trace_df: pd.DataFrame
        The pandas DataFrame for which the memory percentages are computed.
    trace_usage: TraceUsage
        A `TraceUsage` object containing the usage data for the trace.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame obtained from `raw_trace_df` after modifying the
        resource usage statistics to report percentages.

    """
    for mem_col in [specs.AVG_MEM_COL, specs.MAX_MEM_COL]:
        raw_trace_df[mem_col] = pd.Series(raw_trace_df[mem_col].divide(
            trace_usage.get_allocated_mem()).replace(np.inf, 0))
    for cpu_col in [specs.AVG_CPU_COL, specs.MAX_CPU_COL]:
        raw_trace_df[cpu_col] = pd.Series(raw_trace_df[cpu_col].divide(
            trace_usage.get_allocated_cpu()).replace(np.inf, 0))
    return raw_trace_df


def get_differenced_trace(data_trace, diff_level):
    """Computes the result of differencing `data_trace` `diff_level` times.

    Parameters
    ----------
    data_trace: np.array
        A numpy array representing the trace to be differenced.
    diff_level: int
        An integer representing the degree of differencing applied to
        `data_trace`.

    Returns
    -------
    np.array
        The numpy array obtained from `data_trace` after applying differencing
        at a degree of `diff_level`

    """
    diff_ts = np.diff(data_trace, diff_level)
    return impute_for_time_series(diff_ts, 0)


def extract_time_series_from_trace(trace_data, series_name):
    """Extracts the time series for `series_name` from `trace_data`.

    Any NaN values in the time series are converted to 0.

    Parameters
    ----------
    trace_data: pd.DataFrame
        A DataFrame representing the data for the trace.
    series_name: str
        A string representing the name of the data representing the time
        series.

    Returns
    -------
    np.array
        A numpy array representing the time series for `series_name` extracted
        from `trace_data`.

    """
    data_trace = trace_data[series_name].to_numpy()
    return impute_for_time_series(data_trace, 0)


def get_total_spare_during_window(allocated, used, win_start,
                                  win_end, agg_window):
    """The total amount of spare resource during [`win_start`, `win_end`).

    Parameters
    ----------
    allocated: float
        A float representing the amount of the resource allocated.
    used: np.array
        A numpy array representing a time series of the amount of the resource
        used at each time point.
    win_start: int
        An integer representing the start index for the window of interest.
    win_end: int
        An integer representing the end index for the window of interest.
    agg_window: int
        The aggregation period applied to the data.

    Returns
    -------
    float
        A float representing the total amount of the resource that is unused
        during the window defined by [`win_start`, `win_end`).

    """
    win_start *= agg_window
    win_end *= agg_window
    spare_ts = np.maximum(allocated - used[win_start:win_end], 0.0)
    return sum(spare_ts)


def output_time_series_list_to_file(time_series_list, output_file):
    """Writes `time_series_list` to `output_file`.

    The time series in `time_series_list` are written to `output_file` with
    one line per list element. Each line is a comma separated list of values
    corresponding to the observations for the time series.

    Parameters
    ----------
    time_series_list: list
        A list of numpy arrays corresponding to the time series being written.
    output_file: str
        A string representing the name of the file being written to.

    Returns
    -------
    None

    Side Effect
    -----------
    The contents of `time_series_list` are written to `output_file`. If the
    file already exists it is overwritten. Otherwise, the file is created.

    """
    with open(output_file, "w") as ts_file:
        f.write("start_usage, ..., end_usage\n")
        for time_series in time_series_list:
            ts_file.write(",".join(time_series))
            ts_file.write("\n")


def get_mean_absolute_scaled_error(actuals, predicteds):
    """The mean absolute scaled error of `predicteds` vs `actuals`.

    The mean absolute scaled error is calculated as the mean of the scaled
    errors. The scaled errors are the prediction errors scaled by the average
    baseline error. The prediction errors are `|a[i] - p[i]|` where `a[i]` is
    the actual value and `p[i]` is the predicted value for period `i`. The
    baseline errors are the errors from a model that predicts the current value
    based on the past value.

    Parameters
    ----------
    actuals: np.array
        A numpy array representing the actual values.
    predicteds: np.array
        A numpy array representing the predicted values.

    Returns
    -------
    float
        A float representing the mean absolute scaled error between
        `actuals` and `predicteds`.

    """
    actuals = list(actuals)
    predicteds = list(predicteds)
    errors = [np.abs(actuals[i] - predicteds[i]) for i in range(len(actuals))]
    denom = np.mean(np.abs(np.diff(actuals)))
    return np.mean(errors / denom) if denom != 0 else np.mean(errors)

def get_one_sided_errors(actuals, predicteds, lower=True):
    """Gets the one sided error of `actuals` vs `predicted` based on `lower`.

    The one sided errors are the maximum of `actuals - predicteds` and 0 if
    `lower` is true. Otherwise, they are the maximum of `predicteds - actuals`
    and 0.

    Parameters
    ----------
    actuals: list
        The actual values.
    predicteds: list
        The predicted values.
    lower: bool
        A boolean indicating the type of errors computed. If True, the under
        predictions are calculated. Otherwise, the over prediction are
        calculated.

    Returns
    -------
    list
        A list containing the one sided errors.

    """
    if lower:
        return [max(0, actuals[i] - predicteds[i])
                for i in range(len(actuals))]
    return [max(predicteds[i] - actuals[i], 0) for i in range(len(actuals))]


def get_one_sided_mean_absolute_scaled_error(actuals, predicteds, lower=True):
    """The one-sided mean absolute scaled error of `predicteds` vs `actuals`.

    The mean absolute scaled error is calculated as the mean of the scaled
    errors. The scaled errors are the prediction errors scaled by the average
    baseline error. The prediction errors are `|a[i] - p[i]|` where `a[i]` is
    the actual value and `p[i]` is the predicted value for period `i`. The
    baseline errors are the errors from a model that predicts the current
    value based on the past value. The one sided variant only considers under
    or over predictions, with the other errors being set to 0.

    Parameters
    ----------
    actuals: np.array
        A numpy array representing the actual values.
    predicteds: np.array
        A numpy array representing the predicted values.
    lower: bool
        A boolean indicating the type of errors computed. If True, the under
        predictions are calculated. Otherwise, the over prediction are
        calculated.

    Returns
    -------
    float
        A float representing the one-sided mean absolute scaled error between
        `actuals` and `predicteds` based on `lower`.

    """
    actuals = list(actuals)
    predicteds = list(predicteds)
    errors = get_one_sided_errors(actuals, predicteds, lower)
    denom = np.mean(np.abs(np.diff(actuals)))
    return np.mean(errors / denom) if denom != 0 else np.mean(errors)


def calculate_train_and_test_mase(y_train, preds_train, y_test, preds_test):
    """The mean absolute scaled error for the training and testing sets.

    Parameters
    ----------
    y_train: np.array
        A numpy array representing actual values of the target for the
        training set.
    preds_train: np.array
        A numpy array representing predicted values of the target for the
        training set.
    y_test: np.array
        A numpy array representing actual values of the target for the
        testing set.
    preds_test: np.array
        A numpy array representing predicted values of the target for the
        testing set.

    Returns
    -------
    float, float
        Two floats representing the mean absolute scaled errors for the
        training and testing sets, respectively.

    """
    train_mase = get_mean_absolute_scaled_error(y_train, preds_train)
    test_mase = get_mean_absolute_scaled_error(y_test, preds_test)
    return train_mase, test_mase


def get_under_pred_vals(under_preds, pred_count):
    """Gets the proportion and maximum value of `under_preds`.

    Parameters
    ----------
    under_preds: np.array
        A numpy array containing the under predictions.
    pred_count: int
        An integer representing the number of predictions.

    Returns
    -------
    float, float
        A float representing the proportion of predictions that were under
        predictions and a float representing the magnitude of the maximum
        under prediction.

    """
    count = len(under_preds)
    if count == 0:
        return 0.0, np.nan
    return (count / pred_count), max(under_preds)

def get_under_predictions(actuals, predicteds):
    """Retrieves the under predictions of `predicteds` vs `actuals`.

    The under predictions are the actual values minus the predicted values
    divided by the predicted values for the predictions that are less than
    the actuals.

    Parameters
    ----------
    actuals: np.array
        A numpy array representing the actual values.
    predicteds: np.array
        A numpy array representing the actual values.

    Returns
    -------
    list
        A list containing the under predictions.

    """
    return [(actuals[i] - predicteds[i]) / max(predicteds[i], specs.EPS)
            for i in range(len(actuals)) if actuals[i] > predicteds[i]]

def get_under_prediction_stats(actuals, predicteds):
    """Retrieves statistics of under predictions of predicteds vs actuals.

    The proportion and maximum of under predictions are calculated.

    Parameters
    ----------
    actuals: np.array
        A numpy array representing the actual values.
    predicteds: np.array
        A numpy array representing the actual values.

    Returns
    -------
    float, float, float
        A float representing the one-sided mean absolute scaled error for
        under predictions. A float representing the proportion of predictions
        that were under predictions and a float representing the magnitude of
        the maximum under prediction, standardized by the average prediction.

    """
    under_preds = np.array(get_under_predictions(actuals, predicteds))
    under_mase = get_one_sided_mean_absolute_scaled_error(
        actuals, predicteds, True)
    under_prop, under_max = get_under_pred_vals(
        under_preds, len(predicteds))
    return under_mase, under_prop, under_max

def get_over_pred_vals(over_preds, pred_count):
    """Gets the proportion and average value of `over_preds`.

    Parameters
    ----------
    over_preds: np.array
        A numpy array containing the over predictions.
    pred_count: int
        An integer representing the number of predictions.

    Returns
    -------
    float, float
        A float representing the proportion of predictions that were over
        predictions and a float representing the magnitude of the average
        over prediction.

    """
    count = len(over_preds)
    if count == 0:
        return 0.0, 0.0
    return (count / pred_count), np.mean(over_preds)

def get_over_predictions(actuals, predicteds):
    """Retrieves the over predictions of `predicteds` vs `actuals`.

    The under predictions are the predicted values minus the actual values
    divided by the predicted values for the predictions that are greater than
    the actuals.

    Parameters
    ----------
    actuals: np.array
        A numpy array representing the actual values.
    predicteds: np.array
        A numpy array representing the actual values.

    Returns
    -------
    list
        A list containing the under predictions.

    """
    return [(predicteds[i] - actuals[i]) / max(predicteds[i], specs.EPS)
            for i in range(len(actuals)) if actuals[i] < predicteds[i]]

def get_over_prediction_stats(actuals, predicteds):
    """Retrieves statistics of over predictions of predicteds vs actuals.

    The proportion and standardized average of over predictions are
    calculated.

    Parameters
    ----------
    actuals: np.array
        A numpy array representing the actual values.
    predicteds: np.array
        A numpy array representing the actual values.

    Returns
    -------
    float, float, float
        A float representing the one-sided mean absolute scaled error for
        over predictions. A float representing the proportion of predictions
        that were over predictions and a float representing the magnitude of
        the average over prediction, standardized by the average prediction.

    """
    over_preds = np.array(get_over_predictions(actuals, predicteds))
    over_mase = get_one_sided_mean_absolute_scaled_error(
        actuals, predicteds, False)
    over_prop, over_avg = get_over_pred_vals(
        over_preds, len(predicteds))
    return over_mase, over_prop, over_avg

def calculate_evaluation_metrics(y_train, preds_train,
                                 y_test, preds_test, total_spare):
    """Calculates the evaluation metrics for the training and testing sets.

    The evaluation metrics are the mean absolute scaled error for the training
    and testing sets, the one-sided mean absolute scaled error for under
    predictions, the proportion of under predictions, the magnitude of the
    maximum under prediction, the one-sided mean absolute scaled error for
    over predictions, the proportion of over predictions, the magnitude of
    the average over prediction, and the total spare amount of the resource
    over the testing preiod.

    Parameters
    ----------
    y_train: np.array
        A numpy array representing actual values of the target for the
        training set.
    preds_train: np.array
        A numpy array representing predicted values of the target for the
        training set.
    y_test: np.array
        A numpy array representing actual values of the target for the
        testing set.
    preds_test: np.array
        A numpy array representing predicted values of the target for the
        testing set.
    total_spare: float
        A float representing the total spare amount of the resource over the
        test period.

    Returns
    -------
    dict
        A dictionary containing the evaluation metrics. Keys are strings
        representing the names of the evaluation metric and the corresponding
        value is a float.

    """
    train_mase, test_mase = calculate_train_and_test_mase(
        y_train, preds_train, y_test, preds_test)
    under_mase, prop_under_preds, max_under_pred = get_under_prediction_stats(
        list(y_test), list(preds_test))
    over_mase, prop_over_preds, avg_over_pred = get_over_prediction_stats(
        list(y_test), list(preds_test))
    return {"train_mase": train_mase, "test_mase": test_mase,
            "under_mase": under_mase, "prop_under_preds": prop_under_preds,
            "max_under_pred": max_under_pred, "over_mase": over_mase,
            "prop_over_preds": prop_over_preds,
            "avg_over_pred": avg_over_pred, "total_spare": total_spare}


def process_model_results_df(model_results_df):
    """Processes `model_results_df`.

    `model_results_df` is processed by renaming columns in order to compare
    the model results against other model results.

    Parameters
    ----------
    model_results_df: pd.DataFrame
        A pandas DataFrame containing the results of the modeling process
        for each trace.

    Returns
    -------
    pd.DataFrame
        The pandas DataFrame obtained from `model_results_df` after
        processing.

    """
    processed_df = copy.deepcopy(model_results_df)
    cols = ["id"] + ["_".join(col.split("_")[:-1]) for col
                     in processed_df.columns if col != "id"]
    processed_df.columns = cols
    return processed_df


def calculate_harvest_stats(avail_val, actual_ts, predicted_ts, buffer_pct):
    """Calculates the harvest statistics of predicted versus actual values.

    The harvest statistics are the amount of spare resources harvested and
    the number of violations. A violation occurs when the predicted value
    from `predicted_vals` multiplied by `1 + buffer_pct` is still less than
    the actual value. Otherwise, the resource is considered to be harvested
    based on this predicted value. The amount harvested is the difference
    between `avail_val` and the predictions (after adding the buffer) summed
    over all time points in which the predicted value is greater than or
    equal to the actual value.

    Parameters
    ----------
    avail_val: float
        A float representing the amount of the resource allocated to the
        trace for its duration.
    actual_ts: np.array
        A numpy array of actual values for the trace, representing the
        percent of the available resource used at each time point.
    predicted_ts: np.array
        A numpy array of predicted values for the trace, representing the
        percent of the available resource predicted to be used at each
        time point.
    buffer_pct: float
        A float representing the percentage of each prediction to be used as
        a buffer. Thus, the effective prediction for each time period is the
        original prediction multiplied by `1 + buffer_pct`.

    Returns
    -------
    float, int
        A float representing the amount harvested and an integer representing
        the number of violations.

    """
    actuals = avail_val * actual_ts
    predicteds = np.minimum(predicted_ts * avail_val * (1.0 + buffer_pct),
                            avail_val)
    under_pred_indices = predicteds < actuals
    predicted_spare_ts = avail_val - predicteds
    predicted_spare_ts[under_pred_indices] = 0.0
    return np.sum(predicted_spare_ts), np.sum(under_pred_indices)


def calculate_utilization_percent(alloced_amt, usage_ts):
    """Calculates the percent utilization for a trace.

    The percent utilization is the total amount used over the life of the
    trace, `usage_ts`, divided by the total amount allocated, `alloced_amt`,
    over the life of the trace.

    Parameters
    ----------
    alloced_amt: float
        A float representing the amount of the resource allocated to the
        trace at any given time point.
    usage_ts: np.array
        A numpy array representring the absolute value of utilization at each
        time point for the trace.

    Returns
    -------
    float
        A float representing the percent utilization over the life of the
        trace.

    """
    utilization_rates = usage_ts / alloced_amt
    return min(np.mean(utilization_rates), 1.0)
