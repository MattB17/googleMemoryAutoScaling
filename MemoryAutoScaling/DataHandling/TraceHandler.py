"""The `TraceHandler` class is used to perform preprocessing on trace files
generated from the raw google data.

"""
import os
import re
import sys
import io
import pandas as pd
from MemoryAutoScaling import parallel, utils, specs
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
    min_length: int
        An integer representing the minimum length of a time series for a
        trace to be considered.
    agg_window: int
        An integer representing the aggregation window for the traces. That is,
        for each trace, the time series data will be aggregated every
        `agg_window` time periods.

    Attributes
    ----------
    _dir_path: str
        The path to the directory containing the raw data.
    _file_identifier: str
        Identifies the files in `_dir_path` to be processed.
    _min_length: int
        The minimum required length for a time series as part of a trace.
    _agg_window: int
        The aggregation window used for traces.
    _traces: list
        A list of `Trace` objects representing the traces that have been
        processed by the handler.

    """
    def __init__(self, dir_path, file_identifier, min_length, agg_window):
        self._dir_path = dir_path
        self._file_identifier = file_identifier
        self._min_length = min_length
        self._agg_window = agg_window
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
        trace_files = [os.path.join(self._dir_path, trace_file)
                       for trace_file in self.get_trace_files()]
        self._traces = parallel.build_all_traces_from_files(
            trace_files, self._min_length, self._agg_window)

    def run_processing_pipeline(self, verbose=True):
        """Runs the processing pipeline.

        The processing pipeline is run with status updates printed to the
        screen if the `verbose` flag is set to True. Otherwise, these are
        suppressed.

        Parameters
        ----------
        verbose: bool
            A boolean indicating whether progress should be printed. The
            default value is True.

        Returns
        -------
        list
            A list of `Trace` objects representing the traces processed in
            the pipeline.

        """
        if not verbose:
            old_stdout = sys.stdout
            consumed_output = io.StringIO()
            sys.stdout = consumed_output
        traces = self._run_processing_pipeline()
        if not verbose:
            sys.stdout = old_stdout
        return traces

    def _run_processing_pipeline(self):
        """Runs the processing pipeline and prints progress.

        The processing pipeline reads and processes all trace files from
        the input directory and outputs a list of these `Trace` objects.

        Returns
        -------
        list
            A list of `Trace` objects representing the traces processed in
            the pipeline.

        """
        print("Processing Traces")
        print("-----------------")
        self.process_all_trace_files()
        print("Processing Complete")
        print()
        return self.get_traces()

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
        agg_window = file_name_comps[5]
        ts_data = pd.read_csv(ts_file)
        if len(ts_data) >= self._min_length:
            self._traces.append(
                Trace(trace_id, start_time, end_time, ts_data, agg_window))

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
