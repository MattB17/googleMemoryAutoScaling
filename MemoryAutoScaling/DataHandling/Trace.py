"""The `Trace` class stores the data associated with a Google trace from
the Borg cluster.

"""
import pandas as pd
from MemoryAutoScaling import utils
from MemoryAutoScaling import specs


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

    Attributes
    ----------
    _trace_id: int
        Represents the id of the trace.
    _start_time: int
        The time at which the trace data starts.
    _end_time: int
        The time at which the trace data ends.
    _avg_mem_ts: np.array
        A time series representing average memory usage.
    _max_mem_ts: np.array
        A time series representing maximum memory usage.
    _avg_cpu_ts: np.array
        A time series representing average CPU usage.
    _max_cpu_ts: np.array
        A time series representing maximum CPU usage.

    """
    def __init__(self, trace_id, start_time, end_time, ts_df):
        self._trace_id = trace_id
        self._start_time = start_time
        self._end_time = end_time
        self._avg_mem_ts = utils.extract_time_series_from_trace(
            trace_df, specs.AVG_MEM_COL)
        self._max_mem_ts = utils.extract_time_series_from_trace(
            trace_df, specs.MAX_MEM_COL)
        self._avg_cpu_ts = utils.extract_time_series_from_trace(
            trace_df, specs.AVG_CPU_COL)
        self._max_cpu_ts = utils.extract_time_series_from_trace(
            trace_df, specs.MAX_CPU_COL)

    @classmethod
    def from_raw_trace_data(cls, trace_df):
        """Constructs a `Trace` from the raw data in `trace_df`.

        Parameters
        ----------
        trace_df: pd.DataFrame
            A dataframe representing the raw data extracted for a Borg job.

        Returns
        -------
        Trace
            The `Trace` represented by trace_df
        """
        trace_id = int(trace_df[specs.TRACE_ID_COL][0])
        start_time = int(trace_df[specs.START_INTERVAL_COL][0])
        end_time = int(trace_df[specs.END_INTERVAL_COL][-1])
        ts_df = trace_df[[specs.AVG_MEM_COL, specs.MAX_MEM_COL,
                          specs.AVG_CPU_COL, specs.MAX_CPU_COL]]
        return cls(trace_id, start_time, end_time, ts_df)

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

    def get_average_memory_time_series(self):
        """The average memory usage time series for the trace.

        Returns
        -------
        np.array
            A numpy array representing the time series of average memory
            usage.

        """
        return self._avg_mem_ts

    def get_maximum_memory_time_series(self):
        """The maximum memory usage time series for the trace.

        Returns
        -------
        np.array
            A numpy array representing the time series of maximum memory
            usage.

        """
        return self._max_mem_ts

    def get_average_cpu_time_series(self):
        """The average CPU usage time series for the trace.

        Returns
        -------
        np.array
            A numpy array representing the time series of average CPU usage.

        """
        return self._avg_cpu_ts

    def get_maximum_cpu_time_series(self):
        """The maximum CPU usage time series for the trace.

        Returns
        -------
        np.array
            A numpy array representing the time series of maximum CPU usage.

        """
        return self._max_cpu_ts

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
        file_name = "trace_df_{0}_{1}_{2}.csv".format(
            self._trace_id, self._start_time, self._end_time)
        trace_df = pd.DataFrame({specs.AVG_MEM_COL: self._avg_mem_ts,
                                 specs.AVG_CPU_COL: self._avg_cpu_ts,
                                 specs.MAX_MEM_COL: self._max_mem_ts,
                                 specs.MAX_CPU_COL: self._max_cpu_ts})
        trace_df.to_csv(
            os.path.join(output_dir, file_name), sep=",", index=False)
