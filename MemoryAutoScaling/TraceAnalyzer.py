"""The `TraceAnalyzer` class is used to calculate statistics and perform
analysis on a trace.

"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from MemoryAutoScaling import utils


class TraceAnalyzer:
    """Performs analysis on a data trace.

    Parameters
    ----------
    seaborn_style: str
        A string representing the style used for seaborn plots.
    matplotlib_style: str
        A string representing the style used for matplotlib plots.
    tick_interval: int
        An integer representing the interval at which ticks are displayed
        on plots.
    plot_color: str
        A string representing the color for plots
    analysis_title: str
        A string representing the title for the analysis being run.

    Attributes
    ----------
    _tick_interval: int
        The interval at which ticks are displayed on plots.
    _plot_color: str
        The color used for analysis plots.
    _analysis_title: str
        The title of the analysis being run.

    """
    def __init__(self, seaborn_style, matplotlib_style,
                 tick_interval, plot_color, analysis_title):
        sns.set_style(seaborn_style)
        plt.style.use(matplotlib_style)
        self._tick_interval = tick_interval
        self._plot_color = plot_color
        self._analysis_title = analysis_title

    def get_tick_interval(self):
        """Retrieves the tick interval used for plotting.

        Returns
        -------
        int
            An integer representing the interval at which ticks are displayed
            on plots generated by the analyzer.

        """
        return self._tick_interval

    def get_plot_color(self):
        """Retrieves the color used for plotting

        Returns
        -------
        str
            A string representing the color of plots produced by the analyzer.

        """
        return self._plot_color

    def get_analysis_title(self):
        """Retrieves the title for the analysis.

        Returns
        -------
        str
            A string representing the analysis title.

        """
        return self._analysis_title

    def set_tick_interval(self, new_interval):
        """Sets the tick interval to `new_interval`.

        Parameters
        ----------
        new_interval: int
            An integer representing the new interval at which ticks will be
            displayed on plots generated by the analyzer.

        Returns
        -------
        None

        """
        self._tick_interval = new_interval

    def set_plot_color(self, new_color):
        """Sets the plot color to `new_color`.

        Parameters
        ----------
        new_color: str
            A string representing the new color used for plots generated by
            the analyzer.

        Returns
        -------
        None

        """
        self._plot_color = new_color

    def set_analysis_title(self, new_title):
        """Sets the analysis title to `new_title`.

        Parameters
        ----------
        new_title: str
            A string representing the new title for the analysis.

        Returns
        -------
        None

        """
        self._analysis_title = new_title

    def plot_trace(self, data_trace):
        """Plots `data_trace` across time.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the trace to be plotted.

        Returns
        -------
        None

        """
        plt.figure(figsize=(20, 5))
        plt.plot(data_trace, color=self._plot_color, linewidth=3)
        utils.setup_trace_plot(len(data_trace), self._tick_interval,
                               "{} Trace".format(self._analysis_title))

    def plot_deviations_from_average(self, data_trace):
        """Plots the deviations of `data_trace` from its average across time.

        The average value of `data_trace` is computed and subtracted from each
        time point to get a trace recording the deviations of `data_trace`
        from its average. The resulting trace is then plotted.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the raw data trace for which the
            deviations from its average will be plotted.

        Returns
        -------
        None

        """
        plt.figure(figsize=(20, 5))
        plt.plot(np.zeros(len(data_trace)), color="red", linewidth=1)
        plt.plot(data_trace - np.mean(data_trace),
                 color=self._plot_color, linewidth=3)
        utils.setup_trace_plot(len(data_trace), self._tick_interval,
                               "{} Trace - Deviations From Average".format(
                                    self._analysis_title))

    def calculate_statistics(self, data_trace):
        """Calculates statistics for `data_trace`.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing a data trace.

        Returns
        -------
        pd.Series
            A pandas Series containing the statistics for `data_trace`.

        """
        stats = utils.get_trace_stats(data_trace)
        trace_stats = {"std": stats["std"],
                       "range": stats["max"] - stats["min"],
                       "IQR": stats["p75"] - stats["p25"],
                       "median": stats["median"],
                       "avg": stats["avg"]}
        return pd.Series(trace_stats)
