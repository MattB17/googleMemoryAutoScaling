"""A set of utility functions used throughout the memory auto scaling code.

"""
import copy
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
from MemoryAutoScaling import specs


def get_train_cutoff(data_trace, train_prop):
    """Calculates the index identifying the end of the training set.

    That is, an index of `data_trace` is calculated based on `train_prop`
    which identifies the training dataset of `data_trace`.

    Paraneters
    ----------
    data_trace: np.array
        A numpy array representing the data trace.
    train_prop: float
        A float in the range [0, 1] representing the proportion of data
        in the training set.

    Returns
    -------
    int
        An integer representing the index of `data_trace` at which to cutoff for
        the training set.

    """
    return int(len(data_trace) * train_prop)


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
    max_val = np.max(data_trace)
    return {"avg": np.mean(data_trace),
            "std": np.std(data_trace),
            "median": np.median(data_trace),
            "max": np.max(data_trace),
            "range": max_val - np.min(data_trace)}


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


def build_trace_data_from_trace_df(raw_trace_df, agg_window):
    """Builds data for a `Trace` object from `raw_trace_df`.

    The DataFrame containing the data for a `Trace` is built. This is done
    by first replacing NaN values with 0 and then computing the aggregated
    statistics for `raw_trace_df` based on `agg_window`.

    Parameters
    ----------
    trace_df: pd.DataFrame
        A pandas DataFrame containing the raw data for the trace.
    agg_window: int
        An integer representing the aggregation period.

    Returns
    -------
    pd.DataFrame
        The DataFrame obtained from `raw_trace_df` after removing NaN values
        and aggregating data according to `agg_window`.

    """
    trace_df = raw_trace_df[specs.RAW_TIME_SERIES_COLS].fillna(0)
    return build_trace_aggregate_df(trace_df, agg_window)


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

def calculate_evaluation_metrics(y_train, preds_train, y_test, preds_test):
    """Calculates the evaluation metrics for the training and testing sets.

    The evaluation metrics consist of the mean absolute scaled error for
    the training set, the mean absolute scaled error for the testing set,
    the number of under predictions for the testing set, and the magnitude of
    the maximum under prediction for the testing set.

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
    dict
        A dictionary containing the evaluation metrics. Keys are strings
        representing the names of the evaluation metric and the corresponding
        value is a float. These metrics are the mean absolute scaled error
        for the training and testing sets, the one-sided mean absolute scaled
        error for under predictions, the proportion of under predictions, the
        magnitude of the maximum under prediction, the one-sided mean absolute
        scaled error for over predictions, the proportion of over predictions,
        and the magnitude of the average over prediction.

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
            "prop_over_preds": prop_over_preds, "avg_over_pred": avg_over_pred}


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
