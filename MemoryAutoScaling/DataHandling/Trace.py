"""The `Trace` class stores the data associated with a Google trace from
the Borg cluster.

"""
import numpy as np
import pandas as pd
from MemoryAutoScaling import specs, utils


class Trace:
    """Serves as a container for the data relating to a trace.

    Parameters
    ----------
    trace_id: int
        The ID for the trace.
    start_time: int
        The time at which the trace data begins
    end_time: int
        The time at which the trace data ends
    ts_df: pd.DataFrame
        A pandas DataFrame containing the time series data for the trace.
    trace_usage: TraceUsage
        A `TraceUsage` object representing usage information for the trace.
    agg_window: int
        An integer representing the length of the aggregation window. That is,
        for every `agg_window` periods the trace data is aggregated.

    Attributes
    ----------
    _trace_id: int
        Represents the id of the trace.
    _start_time: int
        The time at which the trace data starts.
    _end_time: int
        The time at which the trace data ends.
    _trace_df: pd.DataFrame
        A pandas DataFrame containing the time series data for the trace.
    _trace_usage: TraceUsage
        The usage information for the trace.
    _agg_window: int
        The aggregation window for the trace.

    """
    def __init__(self, trace_id, start_time, end_time,
                 ts_df, trace_usage, agg_window):
        self._trace_id = trace_id
        self._start_time = start_time
        self._end_time = end_time
        self._trace_df = ts_df
        self._trace_usage = trace_usage
        self._agg_window = agg_window
        print("Trace {} created".format(trace_id))

    @classmethod
    def from_raw_trace_data(cls, trace_df, agg_window):
        """Constructs a `Trace` from the raw data in `trace_df`.

        Parameters
        ----------
        trace_df: pd.DataFrame
            A dataframe representing the raw data extracted for a Borg job.
        agg_window: int
            An integer representing the aggregation window for the trace.

        Returns
        -------
        Trace
            The `Trace` represented by trace_df.

        """
        trace_id = int(trace_df[specs.TRACE_ID_COL].to_numpy()[0])
        start_time = int(trace_df[specs.START_INTERVAL_COL].to_numpy()[0])
        end_time = int(trace_df[specs.END_INTERVAL_COL].to_numpy()[-1])
        trace_usage = TraceUsage.from_trace_df(trace_df)
        ts_df = utils.build_trace_data_from_trace_df(
            trace_df, agg_window, trace_usage)
        return cls(
            trace_id, start_time, end_time, ts_df, trace_usage, agg_window)

    def get_trace_id(self):
        """The ID of the trace.

        Returns
        -------
        int
            An integer representing the ID of the trace.

        """
        return self._trace_id

    def get_start_time(self):
        """The time at which the trace starts.

        Returns
        -------
        int
            An integer representing the time point marking the start of the
            trace.

        """
        return self._start_time

    def get_end_time(self):
        """The time at which the trace ends.

        Returns
        -------
        int
            An integer representing the time point marking the end of the
            trace.

        """
        return self._end_time

    def get_aggregation_window(self):
        """The aggregation window used for the trace.

        Returns
        -------
        int
            An integer representing the aggregation window used for the trace.

        """
        return self._agg_window

    def get_maximum_memory_time_series(self):
        """The maximum memory usage time series for the trace.

        Returns
        -------
        np.array
            A numpy array representing the time series of maximum memory
            usage.

        """
        max_mem_col = "{}_ts".format(specs.MAX_MEM_COL)
        return self._trace_df[max_mem_col].values

    def get_spare_memory_in_window(self, win_start, win_end):
        """The spare memory in the interval from [`win_start`, `win_end`).

        The spare memory at a particular time point is the difference between
        the assigned memory and the used memory for that time point. The spare
        for the window is the sum of these values at each point in the window,
        excluding `win_end`.

        Parameters
        ----------
        win_start: int
            An integer representing the start index for the window.
        win_end: int
            An integer representing the end index for the window.

        Returns
        -------
        float
            A float representing the total spare memory for the window.

        """
        return utils.get_total_spare_during_window(
            self._mem_alloc,
            self._mem_alloc * self.get_maximum_memory_time_series(),
            win_start, win_end)

    def get_maximum_cpu_time_series(self):
        """The maximum CPU usage time series for the trace.

        Returns
        -------
        np.array
            A numpy array representing the time series of maximum CPU usage.

        """
        max_cpu_col = "{}_ts".format(specs.MAX_CPU_COL)
        return self._trace_df[max_cpu_col].values

    def get_spare_cpu_in_window(self, win_start, win_end):
        """The spare CPU units in the interval from [`win_start`, `win_end`).

        The spare CPU units at a particular time point is the difference
        between the assigned CPU and the used CPU for that time point. The
        spare for the window is the sum of these values at each point in the
        window, excluding `win_end`.

        Parameters
        ----------
        win_start: int
            An integer representing the start index for the window.
        win_end: int
            An integer representing the end index for the window.

        Returns
        -------
        float
            A float representing the total spare CPU units for the window.

        """
        return utils.get_total_spare_during_window(
            self._cpu_alloc,
            self._cpu_alloc * self.get_maximum_cpu_time_series(),
            win_start, win_end)

    def get_target_time_series(self, target_col):
        """Retrieves the time series based on `target_col`.

        Parameters
        ----------
        target_col: str
            A string representing the name of the target variable for which
            the time series is retrieved from the trace.

        Returns
        -------
        np.array
            A numpy array representing the time series of `target_col`.

        """
        if target_col in [specs.MAX_MEM_COL, specs.MAX_MEM_TS]:
            return self.get_maximum_memory_time_series()
        return self.get_maximum_cpu_time_series()

    def get_amount_allocated_for_target(self, target_col):
        """Retrieves the availability time series for `target_col`.

        Parameters
        ----------
        target_col: str
            A string representing the name of the target variable for which
            the time series is retrieved from the trace.

        Returns
        -------
        float
            A float representing the amount of resource allocated for
            `target_col` for the trace.

        """
        if target_col in [specs.MAX_MEM_COL, specs.MAX_MEM_TS]:
            return self._mem_alloc
        return self._cpu_alloc

    def get_spare_resource_in_window(self, target_col, win_start, win_end):
        """The spare amount of `target_col` in [`win_start`, `win_end`].

        The spare amount of `target_col` at a particular time point is the
        amount available minus the amount used at that time point. The total
        spare for the window is calculated as the sum of these values over all
        time points in the window, excluding `win_end`.

        Parameters
        ----------
        target_col: str
            A string identifying the resource of interest, either memory or
            CPU units.
        win_start: int
            An integer representing the start index of the window.
        win_end: int
            An integer representing the end index of the window.

        Returns
        -------
        float
            A float representing the total spare amount of `target_col` for
            the window.

        """
        if target_col in [specs.MAX_MEM_COL, specs.MAX_MEM_TS]:
            return self.get_spare_memory_in_window(win_start, win_end)
        return self.get_spare_cpu_in_window(win_start, win_end)


    def get_number_of_observations(self):
        """The number of observations of the trace.

        Returns
        -------
        int
            An integer representing the number of observations present
            in the trace.

        """
        return len(self._trace_df)

    def get_trace_df(self):
        """A dataframe for the trace.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the time series for the trace with
            one column per series.

        """
        return self._trace_df[specs.get_trace_columns()]

    def get_max_time_series_df(self):
        """A dataframe of only the maximum usage time series.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the maximum usage time series for
            the trace.

        """
        return self._trace_df[[specs.MAX_MEM_TS, specs.MAX_CPU_TS]]

    def output_trace(self, output_dir):
        """Outputs the trace to a csv file.

        Parameters
        ----------
        output_dir: str
            A string representing the directory to which the trace will be
            written to.

        Returns
        -------
        None

        Side Effect
        -----------
        Outputs the trace to `output_dir` as a csv file.

        """
        file_name = "trace_df_{0}_{1}_{2}_{3}.csv".format(
            self._trace_id, self._start_time,
            self._end_time, self._agg_window)
        trace_df = self.get_trace_df()
        trace_df.to_csv(
            os.path.join(output_dir, file_name), sep=",", index=False)

    def get_lagged_df(self, lags):
        """Gets the lagged dataframe for the trace at lags of `lags`.

        The lagged dataframe consists of the original columns plus these
        same columns after lagging them by a period of time where the lags
        are equal to the elements of `lags`.

        Parameters
        ----------
        lags: list
            A list of integers representing the lags used to generate
            the dataframe.

        Returns
        -------
        pd.DataFrame
            The lagged DataFrame in which the columns are lagged by the time
            periods indicated in `lags`.

        """
        max_lag = max(lags)
        trace_df = self.get_trace_df()
        target_data = trace_df[max_lag:]
        lagged_dfs = []
        for lag in lags:
            lagged_data = trace_df[(max_lag - lag):-lag]
            lagged_data.columns = specs.get_lagged_trace_columns([lag])
            lagged_data = lagged_data.reset_index(drop=True)
            lagged_dfs.append(lagged_data)
        target_data = target_data.reset_index(drop=True)

        return pd.concat([target_data] + lagged_dfs, axis=1)

    def __str__(self):
        """Computes a string representation of the trace.

        Returns
        -------
        str
            A string representing the trace.

        """
        return ("Trace {0} - {1} Observations from {2} to {3}\n"
                "Aggregated every {4} time periods.".format(
                    self._trace_id, self.get_number_of_observations(),
                    self._start_time, self._end_time, self._agg_window))
