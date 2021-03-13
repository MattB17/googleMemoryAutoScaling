"""The `TraceHandler` class is used to perform preprocessing on trace files
generated from the raw google data.

"""
import os
import re
import pandas as pd
from MemoryAutoScaling import utils
from MemoryAutoScaling import specs
from MemoryAutoScaling.DataHandling.Trace import Trace


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
    _traces: list
        A list of `Trace` objects representing the traces that have been
        processed by the handler.

    """
    def __init__(self, dir_path, file_identifier):
        self._dir_path = dir_path
        self._file_identifier = file_identifier
        self._traces = []

    def get_directory_path(self):
        """The directory from which trace files are read.

        Returns
        -------
        str
            A string representing the directory from which trace files
            are read.

        """
        return self._dir_path

    def update_directory_path(self, new_dir_path):
        """Updates the read directory to `new_dir_path`.

        That is, `new_dir_path` is now the directory from which trace files
        are read.

        Parameters
        ----------
        new_dir_path: str
            A string representing the path of the new read directory.

        Returns
        -------
        None

        """
        self._dir_path = new_dir_path

    def get_file_identifier(self):
        """The identifier used to denote the trace files to be processed.

        The file identifier only refers to trace files in the read directory.

        Returns
        -------
        str
            A string used to identify the trace files for processing.

        """
        return self._file_identifier

    def update_file_identifier(self, new_identifier):
        """Updates the file identifier to `new_identifier`.

        That is, `new_identifier` is now used to denote the trace files to be
        processed by the handler.

        Parameters
        ----------
        new_identifier: str
            A string representing the new identifier for trace files.

        Returns
        -------
        None

        """
        self._file_identifier = new_identifier

    def get_traces(self):
        """Retrieves the traces processed by the handler.

        Returns
        -------
        list
            A list of `Trace` objects representing the traces that were
            processed by the handler.

        """
        return self._traces

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

    def process_all_trace_files(self):
        """Performs the processing pipeline on all trace files.

        All trace files in the input directory are retrieved and processed
        to turn them into `Trace` objects.

        Returns
        -------
        None

        """
        for trace_file in self.get_trace_files():
            trace_path = os.path.join(self._dir_path, trace_file)
            self.process_raw_trace_file(trace_path)

    def process_raw_trace_file(self, trace_file):
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
        order = trace_df[specs.START_INTERVAL_COL].sort_values().index
        trace_df = trace_df.loc[order]
        self._traces.append(Trace.from_raw_trace_data(trace_df))

    def load_all_traces_from_time_series_files(self):
        """Loads all data traces from the time series files.

        Returns
        -------
        None

        """
        for ts_file in self.get_trace_files():
            ts_path = os.path.join(self._dir_path, ts_file)
            self.load_trace_from_time_series_file(ts_path)

    def load_trace_from_time_series_file(self, ts_file):
        """Loads a data trace from a time series file.

        Parameters
        ----------
        ts_file: str
            A string to a time series file. This file only contains the
            time series data corresponding to the trace.

        Returns
        -------
        None

        """
        file_name_comps = ts_file.split("/")[-1].split(".")[0].split("_")
        trace_id = file_name_comps[2]
        start_time = file_name_comps[3]
        end_time = file_name_comps[4]
        ts_data = pd.read_csv(ts_file)
        self._traces.append(Trace(trace_id, start_time, end_time, ts_data))

    def output_data_traces(self, output_dir):
        """Outputs each trace to its own csv file

        A file is created for each file in `_traces` in `output_dir`

        Returns
        -------
        None

        Side Effect
        -----------
        Writes one file for each data trace to `output_dir`.

        """
        for trace in self._traces:
            trace.output_trace(output_dir)
