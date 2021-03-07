"""The `TraceHandler` class is used to perform preprocessing on trace files
generated from the raw google data.

"""
import pandas as pd


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
