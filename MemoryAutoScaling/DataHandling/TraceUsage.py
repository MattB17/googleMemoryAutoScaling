"""The `TraceUsage` class is a container for raw, unaggregated trace
statistics, which are used to calculate utilization and the amount
of spare units for both CPU and memory.

"""
import numpy as np
import pandas as pd
from MemoryAutoScaling import specs, utils


class TraceUsage:
    """Stores raw, unaggregated trace statistics for usage calculations.

    Parameters
    ----------
    mem_alloc: float
        A float representing the amount of memory allocated to the trace for
        its duration.
    cpu_alloc: float
        A float representing the amount of CPU allocated to the trace for its
        duration.
    avg_mem_ts: np.array
        A numpy array representing a time series of the average memory usage
        of the trace.
    avg_cpu_ts: np.array
        A numpy array representing a time series of the average CPU usage of
        the trace.

    Attributes
    ----------
    _mem_alloc: float
        The amount of memory allocated to the trace for its duration.
    _cpu_alloc: float
        The amount of CPU allocated to the trace for its duration.
    _avg_mem_ts: np.array
        A time series of the average memory usage of the trace.
    _avg_cpu_ts: np.array
        A time series of the average CPU usage of the trace.

    """
    def __init__(self, mem_alloc, cpu_alloc, avg_mem_ts, avg_cpu_ts):
        self._mem_alloc = mem_alloc
        self._cpu_alloc = cpu_alloc
        self._avg_mem_ts = avg_mem_ts
        self._avg_cpu_ts = avg_cpu_ts

    @classmethod
    def from_trace_df(cls, trace_df):
        """Creates a `TraceUsage` object from `trace_df`.

        Parameters
        ----------
        trace_df: pd.DataFrame
            The pandas DataFrame containing the raw trace data.

        Returns
        -------
        TraceUsage
            A `TraceUsage` object based on the data in `trace_df`.

        """
        mem_allocated = utils.calculate_max_allocated(
            trace_df, specs.TOTAL_MEM_COL)
        cpu_allocated = utils.calculate_max_allocated(
            trace_df, specs.MAX_CPU_COL)
        avg_mem = trace_df[specs.AVG_MEM_COL].replace(np.nan, 0).values
        avg_cpu = trace_df[specs.AVG_CPU_COL].replace(np.nan, 0).values
        return cls(mem_allocated, cpu_allocated, avg_mem, avg_cpu)

    def get_allocated_mem(self):
        """The memory allocated to the trace for its duration.

        Returns
        -------
        float
            A float representing the memory allocated to the trace for
            its duration.

        """
        return self._mem_alloc

    def get_allocated_cpu(self):
        """The CPU allocated to the trace for its duration.

        Returns
        -------
        float
            A float representing the CPU allocated to the trace for
            its duration.

        """
        return self._cpu_alloc

    def get_avg_mem_usage(self):
        """A time series of average memory usage for the trace.

        Returns
        -------
        np.array
            A numpy array representing a time series of average memory usage
            for the trace.

        """
        return self._avg_mem_ts

    def get_avg_cpu_usage(self):
        """A time series of average CPU usage for the trace.

        Returns
        -------
        np.array
            A numpy array representing a time series of average memory usage
            for the trace.

        """
        return self._avg_cpu_ts

    def get_spare_mem_in_window(self, win_start, win_end, agg_window):
        """The spare amount of memory in [`win_start`, `win_end`].

        The spare amount of memory at a time point is the difference between
        the amount of memory assigned to the machine and the amount taken by
        the trace at that time point. The spare amount of memory in the window
        is the sum of the spare amount at each time point in
        [`win_start`, `win_end`].

        Parameters
        ----------
        win_start: int
            An integer representing the start index of the window.
        win_end: int
            An integer representing the end index of the window.
        agg_window: int
            The aggregation period for the trace


        Returns
        -------
        float
            A float representing the total amount of spare memory on the
            machine in the window, after accounting for the memory consumed
            by the trace.

        """
        return utils.get_total_spare_during_window(
            self._mem_alloc, self._avg_mem_ts, win_start, win_end, agg_window)

    def get_spare_cpu_in_window(self, win_start, win_end):
        """The spare amount of CPU in [`win_start`, `win_end`].

        The spare amount of CPU at a time point is the difference between
        the amount of CPU assigned to the machine and the amount taken by
        the trace at that time point. The spare amount of CPU in the window is
        the sum of the spare amount at each time point in
        [`win_start`, `win_end`].

        Parameters
        ----------
        win_start: int
            An integer representing the start index of the window.
        win_end: int
            An integer representing the end index of the window.
        agg_window: int
            The aggregation period for the trace


        Returns
        -------
        float
            A float representing the total amount of spare CPU on the
            machine in the window, after accounting for the memory consumed
            by the trace.

        """
        return utils.get_total_spare_during_window(
            self._cpu_alloc, self._avg_cpu_ts, win_start, win_end, agg_window)

    def get_spare_resource_in_window(self, target_col, win_start,
                                     win_end, agg_window):
        """The total spare amount of `target_col` in [`win_start`, `win_end`].

        The spare amount of the resource at a time point is the difference
        between the amount of CPU assigned to the machine and the amount
        taken by the trace at that time point. The total amount of spare is
        the sum of the spare amount at each time point in
        [`win_start`, `win_end`].

        Parameters
        ----------
        target_col: str
            A string identifying the resource of interest, either memory or
            CPU units.
        win_start: int
            An integer representing the start index of the window.
        win_end: int
            An integer representing the end index of the window.
        agg_window: int
            The aggregation period for the trace

        Returns
        -------
        float
            A float representing the total amount of spare CPU on the
            machine in the window, after accounting for the memory consumed
            by the trace.

        """
        if target_col in [specs.MAX_MEM_COL, specs.MAX_MEM_TS]:
            return self.get_spare_mem_in_window(win_start, win_end, agg_window)
        return self.get_spare_cpu_in_window(win_start, win_end, agg_window)

    def get_mem_utilization(self):
        """The memory utilization for the trace.

        The memory utilization for a time point is the average memory
        usage for that time point divided by the memory allocated in
        that period. The total memory utilization is the average memory
        utilization over all time points.

        Returns
        -------
        float
            A float representing total memory utilization for the trace.

        """
        return utils.calculate_utilization_percent(
            self._mem_alloc, self._avg_mem_ts)

    def get_cpu_utilization(self):
        """The CPU utilization for the trace.

        The CPU utilization for a time point is the average CPU usage
        for that time point divided by the CPU units allocated in that
        period. The total CPU utilization is the average CPU utilization
        over all time points.

        Returns
        -------
        float
            A float representing total memory utilization for the trace.

        """
        return utils.calculate_utilization_percent(
            self._cpu_alloc, self._avg_cpu_ts)

    def get_resource_utilization(self, resource_col):
        """The utilization of `resource_col` for the trace.

        The resource utilization for a time point is the average usage of
        `resource_col` for that time point divided by the resource units
        allocated in that period. The total resource utilization is the
        average resource utilization over all time points.

        Parameters
        ----------
        resource_col: str
            A string identifying the resource for which the utilization is
            calculated, either memory or CPU.

        Returns
        -------
        float
            A float representing the utilization rate of `resource_col` for
            the trace.

        """
        if resource_col in [specs.MAX_MEM_COL, specs.MAX_MEM_TS]:
            return self.get_mem_utilization()
        return self.get_cpu_utilization()
