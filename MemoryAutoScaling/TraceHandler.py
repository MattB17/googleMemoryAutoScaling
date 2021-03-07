"""The `TraceHandler` class is used to perform preprocessing on trace files
generated from the raw google data.

"""
import pandas as pd


MAX_MEM_COL = 'maximum_usage.memory'
AVG_MEM_COL = 'average_usage.memory'
TRACE_COLS = ['start_time', 'end_time', 'instance_index',
              'alloc_instance_index', 'collection_type', 'assigned_memory',
              'page_cache_memory', 'cycles_per_instruction',
              'memory_accesses_per_instruction', AVG_MEM_COL, MAX_MEM_COL]


class TraceHandler:
    """Used to process traces from the raw google data.

    The google traces are arranged into csv files with a separate file for
    each trace.

    Parameters
    ----------
    dir_path: str
        A string representing the path to the directory  containing the raw
        trace data.
    file_identifier: str
        A string representing an identifier for each file in `dir_path` that
        corresponds to a trace file to be processed.

    Attributes
    ----------
    _dir_path: str
        The path to the directory containing the raw data.
    _file_identifier: str
        Identifies the files in `_dir_path` to be processed.
    _max_mem_traces: list
        A list containing the traces representing max memory usage
        for each processed trace. That is, for each time point, the trace
        records the max memory usage in that time interval.
    _avg_mem_traces: list
        A list containing the traces representing average memory usage
        for each processed trace. That is, for each time point, the trace
        records the average memory usage in that time interval.

    """
    def __init__(self, dir_path, file_identifier):
        self._dir_path = dir_path
        self._file_identifier = file_identifier
        self._max_mem_traces = []
        self._avg_mem_traces = []

    def get_trace_files(self):
        """Retrieves all trace files for processing.

        Returns
        -------
        list
            A list of strings representing the names of the files containing
            the raw data traces.

        """
        match_str = r".*{}.*\.csv".format(self._file_identifier)
        return [file_name for file_name in os.listdir(self._dir_path)
                if re.match(match_str, file_name)]

    def process_trace_file(self, trace_file):
        """Performs the processing pipeline on `trace_file`.

        Parameters
        ----------
        trace_file: str
            A string representing the name of the file to be processed.

        Returns
        -------
        None

        """
        trace_df = pd.read_csv(trace_file)
        trace_df = trace_df[TRACE_COLS]
        order = trace_df['start_time'].sort_values().index
        trace_df = trace_df.loc[order]
        self._max_mem_traces.append(utils.extract_time_series_from_trace(
            trace_df, MAX_MEM_COL))
        self._avg_mem_traces.append(utils.extract_time_series_from_trace(
            trace_df, AVG_MEM_COL))
        trace_df.to_csv(trace_file, sep=',', index=False)
