"""The `TraceAnalyzer` class is used to calculate statistics and perform
analysis on a trace.

"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from MemoryAutoScaling import utils
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


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
        plt.figure(figsize=(20, 8))
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
        plt.figure(figsize=(20, 8))
        plt.plot(np.zeros(len(data_trace)), color="red", linewidth=1)
        plt.plot(data_trace - np.mean(data_trace),
                 color=self._plot_color, linewidth=3)
        utils.setup_trace_plot(len(data_trace), self._tick_interval,
                               "{} Trace - Deviations From Average".format(
                                    self._analysis_title))

    def plot_auto_correlations(self, data_trace, lags):
        """Plots auto correlations for `data_trace`.

        The auto correlations and partial auto correlations are plotted
        for `data_trace` with a lag of `lags`.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the data trace for which auto
            correlations are plotted.
        lags: int
            The number of lags to include in the auto correlation plots.

        Returns
        -------
        None

        """
        plt.figure(figsize=(20, 8))
        layout = (2, 2)
        ts_axis = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_axis = plt.subplot2grid(layout, (1, 0))
        pacf_axis = plt.subplot2grid(layout, (1, 1))
        utils.plot_autocorrelations_for_data(
            data_trace, ts_axis, acf_axis, pacf_axis, lags, self._plot_color)
        plt.suptitle("{} Trace".format(self._analysis_title))
        plt.show()

    def plot_differenced_auto_correlations(self, data_trace, lags):
        """Plot autocorrelations for `data_trace` and its differencing.

        The autocorrelations and partial autocorrelations are plotted for both
        `data_trace` and the time series generated by taking successive
        differences for `data_trace`. The plots are plotted with lags of
        `lags`.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the data trace for which auto
            correlations are plotted.
        lags: int
            The number of lags to include in the auto correlation plots.

        Returns
        -------
        None

        """
        plt.figure(figsize=(20, 8))
        layout = (3, 2)
        ts_axis = plt.subplot2grid(layout, (0, 0))
        diff_axis = plt.subplot2grid(layout, (0, 1))
        acf_axis = plt.subplot2grid(layout, (1, 0))
        diff_acf_axis = plt.subplot2grid(layout, (1, 1))
        pacf_axis = plt.subplot2grid(layout, (2, 0))
        diff_pacf_axis = plt.subplot2grid(layout, (2, 1))

        utils.plot_autocorrelations_for_data(
            data_trace, ts_axis, acf_axis, pacf_axis, lags, self._plot_color)

        diffs = utils.get_differenced_trace(data_trace, 1)
        utils.plot_autocorrelations_for_data(
            diffs, diff_axis, diff_acf_axis, diff_pacf_axis, lags, "red")

        plt.suptitle(
            "{} Trace vs Differenced Trace".format(self._analysis_title))
        plt.show()

    def plot_differenced_time_series(self, data_trace):
        """Plots `data_trace` and the differences time series.

        The differences time series is a time series obtained from
        `data_trace` by taking successive differences from the observations
        of data traces.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the data trace from which the
            successive differences are calculated.

        Returns
        -------
        None

        """
        plt.figure(figsize=(12, 8))
        plt.plot(data_trace, color=self._plot_color,
                 linewidth=3, label="Raw")
        plt.plot(utils.get_differenced_trace(data_trace, 1),
                 color="red", linewidth=3, label="Differenced")
        plt.legend()
        utils.setup_trace_plot(
            len(data_trace), self._tick_interval,
            "{} Trace And 1 Level Differencing".format(self._analysis_title))


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

    def test_for_stationarity(self, data_trace):
        """Tests for stationarity in the `data_trace` time series.

        The augmented Dickey-Fuller statistic is used to test for
        stationarity.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the time series being tested for
            stationarity.

        Returns
        -------
        float
            A float representing the p-value of the augmented Dickey-Fuller
            test for stationarity applied to `data_trace`.

        """
        try:
            return adfuller(data_trace)[1]
        except:
            return np.nan

    def test_for_causality(self, trace_data, col_names, lags):
        """Tests for causality in `trace_data` between `col_names`.

        The Granger causality test is applied to test whether the time series
        indicated by the second element of `col_names` is useful in
        forecasting the time series indicated by the first element of
        `col_names` based on the data in `trace_data`. The tests are applied
        at the lags indicated by `lags`.

        Parameters
        ----------
        trace_data: pd.DataFrame
            A pandas DataFrame containing time series associated with a data
            trace. There is one time series per column.
        col_names: list
            A list of the columns used in the test. It is a two element list
            where the first element represents the name of the target time
            series and the second element represents the name of the time
            series being tested for whether it is useful in forecasting the
            target time series.
        lags: list
            A list of integers representing the lags used in the causality
            test. For each lag of `lags`, the time series indicated by the
            second element of `col_names` will be tested at that lag to see
            if it causes a change in the target time series.

        Returns
        -------
        dict
            A dictionary indicating the results of the Granger test for
            causality for the data in `trace_data` indicated by `col_names`.

        """
        return grangercausalitytests(trace_data[col_names], lags)
