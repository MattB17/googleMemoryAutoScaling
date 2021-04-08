"""A set of utility functions used throughout the memory auto scaling code.

"""
import copy
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
from MemoryAutoScaling import specs


def setup_trace_plot(time_points, tick_interval, title):
    """Performs setup for plots containing trace data.

    Setup involves formatting x axis labels, adding a title and axis labels,
    and displaying the plot.

    Parameters
    ----------
    time_points: int
        An integer representing the number of time points captured in the
        traces that are being plotted.
    tick_interval: int
        An integer representing the interval to display x axis labels. That is,
        for every `tick_interval` time points, a label is printed.
    title: str
        A string representing the plot title.

    """
    tick_labels = ["t{}".format(time_point) if time_point % tick_interval == 0
                   else "" for time_point in range(time_points)]
    plt.xticks(range(time_points), tick_labels)
    plt.xlabel("Time")
    plt.title(title)
    plt.show()


def plot_trace_and_prediction(actual, preds, title):
    """Plots `actual` and its predictions given by `preds`.

    `actual` represents a real trace and `preds` represent a set of
    predictions for the trace. The two are plotted on the same axis
    to see the accuracy of the predictions.

    Parameters
    ----------
    actual: np.array
        A numpy array representing the data trace being plotted.
    preds: np.array
        A numpy array representing the predictions being plotted
    title: str
        A string representing the title for the plot being produced.

    Returns
    -------
    None

    """
    plt.figure(figsize=(20, 8))
    plt.plot(actual, color="blue", linewidth=3, label="Actual")
    plt.plot(preds, color="red", linewidth=3, label="Predicted")
    plt.legend()
    tick_interval = len(actual) // 30
    setup_trace_plot(len(actual), tick_interval, title)


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


def plot_autocorrelations_for_data(trace, trace_ax, acf_ax,
                                   pacf_ax, lags, color):
    """Plots the autocorrelations for `trace`.

    The raw time series `trace` is plotted as well as its autocorrelation
    and partial autocorrelation based on `lags`.

    Parameters
    ----------
    trace: np.array
        The numpy array representing the data trace.
    trace_ax: plt.axis
        The axis on which the raw time series is plotted.
    acf_ax: plt.axis
        The axis on which the autocorrelations are plotted.
    pacf_ax: plt.axis
        The axis on which the partial autocorrelations are plotted.
    lags: int
        An integer representing the number of lags in the autocorrelation
        plots.
    color: str
        A string representing a color for the plot.

    Returns
    -------
    None

    """
    trace_ax.plot(trace, color=color, linewidth=3)
    smt.graphics.plot_acf(trace, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(trace, lags=lags, ax=pacf_ax)


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
    return {"avg": np.mean(data_trace),
            "std": np.std(data_trace),
            "median": np.median(data_trace),
            "max": np.max(data_trace),
            "min": np.min(data_trace),
            "p25": np.percentile(data_trace, 25),
            "p75": np.percentile(data_trace, 75)}


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


def get_trace_columns():
    """The columns in a trace dataframe.

    Returns
    -------
    list
        A list of strings representing the names of the columns in a
        trace dataframe.

    """
    return [specs.AVG_MEM_COL, specs.AVG_CPU_COL,
            specs.MAX_MEM_COL, specs.MAX_CPU_COL]


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


def get_mean_absolute_percentage_error(actuals, predicteds):
    """The mean absolute percentage error of `predicteds` vs `actuals`.

    The mean absolute percetage error is calculated as the average of
    `|a[i] - p[i]| / a[i]` where `a[i]` and `p[i]` refer to the values of
    `actuals` and `predicteds` at index `i`, respectively. The index `i`
    ranges over the indices of `actuals`. Whenever the value of `a[i]` in
    the denominator is zero it is replaced by the median to avoid division
    by zero errors. The result is multiplied by 100 to get a percentage.

    Parameters
    ----------
    actuals: np.array
        A numpy array representing the actual values.
    predicteds: np.array
        A numpy array representing the predicted values.

    Returns
    -------
    float
        A float representing the mean absolute percentage error between
        `actuals` and `predicteds`.

    """
    med = np.median(actuals)
    denoms = copy.deepcopy(actuals)
    denoms[denoms == 0] = med
    return np.mean(np.abs((actuals - predicteds) / denoms)) * 100


def calculate_train_and_test_mape(y_train, preds_train, y_test, preds_test):
    """The mean absolute percentage error for the training and testing sets.

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
        Two floats representing the mean absolute percentage errors for the
        training and testing sets, respectively.

    """
    train_mape = get_mean_absolute_percentage_error(y_train, preds_train)
    test_mape = get_mean_absolute_percentage_error(y_test, preds_test)
    print(train_mape)
    print(test_mape)
    return train_mape, test_mape

def plot_actual_vs_predicted_on_axis(actual, predicted, ax, ax_title):
    """Plots `actual` vs `predicted` on `ax`.

    Parameters
    ----------
    actual: np.array
        A numpy array representing actual values.
    predicted: np.array
        A numpy array representing predicted values.
    ax: plt.axis
        A matplotlib axis on which the plot will be rendered.
    ax_title: str
        A string representing the title for the axis.

    Returns
    -------
    None

    """
    ax.plot(actual, color="blue", linewidth=3)
    ax.plot(predicted, color="red", linewidth=3)
    ax.set_title(ax_title)

def plot_train_and_test_predictions_on_axes(y_train, preds_train, y_test,
                                            preds_test, axes, title):
    """Plots actual values vs predicted values on the axes given by `axes`.

    The actual vs predicted values are plotted for both the training and
    testing sets on the different axes of `axes`.

    Parameters
    ----------
    y_train: np.array
        A numpy array containing the actual observations for the training set.
    preds_train: np.array
        A numpy array containing the predictions for the training set.
    y_test: np.array
        A numpy array containing the actual observations for the testing set.
    preds_test: np.array
        A numpy array containing the predictions for the testing set.
    axes: tuple
        A tuple containing the axes on which the plots will be rendered.
    title: str
        A string representing the title used for the plots.

    Returns
    -------
    None

    """
    plot_actual_vs_predicted_on_axis(
        y_train, preds_train, axes[0], "{} Training Set".format(title))
    plot_actual_vs_predicted_on_axis(
        y_test, preds_test, axes[1], "{} Testing Set".format(title))
    plt.show()

def get_under_pred_vals(under_preds, pred_count, avg_pred):
    """Gets the proportion and maximum value of `under_preds`.

    Parameters
    ----------
    under_preds: np.array
        A numpy array containing the under predictions.
    pred_count: int
        An integer representing the number of predictions
    avg_pred: float
        The average value of the predictions in the prediction window.

    Returns
    -------
    float, float
        A float representing the proportion of predictions that were under
        predictions and a float representing the magnitude of the maximum
        under prediction divided by `avg_pred`.

    """
    count = len(under_preds)
    if count == 0:
        return 0.0, np.nan
    return (count / pred_count), (max(under_preds) / avg_pred)

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
    float, float
        A float representing the proportion of predictions that were under
        predictions and a float representing the magnitude of the maximum
        under prediction, standardized by the average prediction.

    """
    under_preds = np.array([actuals[i] - predicteds[i]
                            for i in range(len(actuals))
                            if actuals[i] > predicteds[i]])
    return get_under_pred_vals(
        under_preds, len(predicteds), np.mean(predicteds))

def get_over_pred_vals(over_preds, pred_count, avg_pred):
    """Gets the proportion and average value of `over_preds`.

    Parameters
    ----------
    over_preds: np.array
        A numpy array containing the over predictions.
    pred_count: int
        An integer representing the number of predictions.
    avg_pred: float
        The average value of the predictions in the prediction window.

    Returns
    -------
    float, float
        A float representing the proportion of predictions that were over
        predictions and a float representing the magnitude of the average
        over prediction divided by `avg_pred`.

    """
    count = len(over_preds)
    if count == 0:
        return 0.0, 0.0
    return (count / pred_count), (np.mean(over_preds) / avg_pred)

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
    float, float
        A float representing the proportion of predictions that were under
        predictions and a float representing the magnitude of the maximum
        under prediction, standardized by the average prediction.

    """
    over_preds = np.array([predicteds[i] - actuals[i]
                            for i in range(len(actuals))
                            if actuals[i] < predicteds[i]])
    return get_over_pred_vals(
        over_preds, len(predicteds), np.mean(predicteds))

def calculate_evaluation_metrics(y_train, preds_train, y_test, preds_test):
    """Calculates the evaluation metrics for the training and testing sets.

    The evaluation metrics consist of the mean absolute percentage error for
    the training set, the mean absolute percentage error for the testing set,
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
    tuple
        A tuple of six floats. The first two represent the mean absolute
        percentage error for the training and testing sets, respectively.
        The next two represent the proportion of under predictions and the
        magnitude of the maximum under prediction, respectively. The last two
        represent the proportion of over predictions and the magnitude of the
        average over prediction.

    """
    train_mape, test_mape = calculate_train_and_test_mape(
        y_train, preds_train, y_test, preds_test)
    prop_under_preds, max_under_pred = get_under_prediction_stats(
        list(y_test), list(preds_test))
    prop_over_preds, avg_over_pred = get_over_prediction_stats(
        list(y_test), list(preds_test))
    return (train_mape, test_mape, prop_under_preds,
            max_under_pred, prop_over_preds, avg_over_pred)

def render_x_y_plot(x, y, title, color):
    """Renders a plot of `x` vs `y`.

    Parameters
    ----------
    x: np.array
        A numpy array representing the x values for the plot.
    y: np.array
        A numpy array representing the y values for the plot.
    title: str
        A string representing the title for the plot.
    color: str
        A string representing the color used for the plotted data.

    Returns
    -------
    None

    """
    plt.figure(figsize=(20, 8))
    plt.plot(x, y, color=color)
    plt.title(title)
    plt.show()
