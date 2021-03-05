"""A set of utility functions used throughout the memory auto scaling code.

"""
import numpy as np
import matplotlib.pyplot as plt
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
