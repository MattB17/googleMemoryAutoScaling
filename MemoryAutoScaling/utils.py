"""A set of utility functions used throughout the memory auto scaling code.

"""
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import copy


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

def extract_time_series_from_trace(trace_data, series_name):
    """Extracts the time series for `series_name` from `trace_data`.

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
    return trace_data[series_name].to_numpy()

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
