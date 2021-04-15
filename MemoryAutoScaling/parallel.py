"""A set of helper functions used in parallel processing.

"""
import numpy as np
import pandas as pd
import multiprocessing as mp
from MemoryAutoScaling import specs
from MemoryAutoScaling.DataHandling import Trace
from MemoryAutoScaling.Analysis import ModelResults


def get_cores_and_traces_per_core(trace_count):
    """Gets the number of cores to use and the number of traces per core.

    The number of cores to use is determined based on the system and
    `trace_count`. Then, given the number of cores, the number of
    traces to be processed on each core is calculated based on `trace_count`.

    Parameters
    ----------
    trace_count: int
        An integer representing the number of total traces to be processed.

    Returns
    -------
    int, int
        Two integers representing the number of cores to use and the number
        of traces to be handled by each core, respectively.

    """
    core_count = min(trace_count, mp.cpu_count() - 1)
    traces_per_core = int(np.ceil(trace_count / core_count))
    return core_count, traces_per_core

def get_traces_for_core(traces, traces_per_core, core_num):
    """Gets the traces from `traces` for the core specified by `core_num`.

    Subsets `traces` to a list of length `traces_per_core` to get a list of
    traces to be processed on the core specified by `core_num`.

    Parameters
    ----------
    traces: list
        A list of `Trace` objects.
    traces_per_core: int
        An integer specifying the number of traces to be processed by
        each core.
    core_num: int
        An integer representing the specific core processing the subsetted
        traces.

    Returns
    -------
    list
        A list representing the subset of `Trace` objects in `traces` that
        will be processed on the core specified by `core_num`.

    """
    start = traces_per_core * core_num
    end = min(len(traces), traces_per_core * (core_num + 1))
    return traces[start:end]

def initialize_and_join_processes(procs):
    """Initializes and joins all the processes in `procs`.

    Parameters
    ----------
    procs: list
        A list of `mp.Process` objects representing the processes
        to be initialized and joined.

    Returns
    -------
    None

    """
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()

def perform_trace_modelling(traces, model_func, train_prop):
    """Performs the modelling procedure on `traces` according to `model_func`.

    Parameters
    ----------
    traces: list
        A list of `Trace` objects to be modelled
    model_func: function
        A function specifying how modelling should be carried out for a
        collection of traces. The function takes three parameters: a list
        of `Trace` objects, a list to store model results, and a float in the
        range [0, 1] representing the proportion of data in the training set.
    train_prop: float
        A float representing the proportion of data in the training set.

    Returns
    -------
    dict
        A dictionary of lists in which the keys are strings representing the
        id of a trace and the values correspond to model results for the
        trace.

    """
    results = mp.Manager().dict()
    procs = []
    cores, traces_per_core = get_cores_and_traces_per_core(len(traces))
    for core_num in range(cores):
        core_traces = get_traces_for_core(traces, traces_per_core, core_num)
        procs.append(mp.Process(target=model_func,
                                args=(core_traces, results, train_prop)))
    initialize_and_join_processes(procs)
    return dict(results)

def build_traces_from_files(trace_files, traces_lst, min_length, agg_window):
    """Builds a list of traces from `trace_files`.

    Parameters
    ----------
    trace_files: list
        A list of strings representing the files from which the traces
        are built.
    traces_lst: mp.Manager().list
        A multiprocessing list to which traces are appended.
    min_length: int
        An integer representing the minimum number of time points needed
        for a trace to be considered.
    agg_window: int
        An integer representing the aggregation window for the trace.

    Returns
    -------
    None

    """
    for trace_file in trace_files:
        trace_df = pd.read_csv(trace_file)
        order = trace_df[specs.START_INTERVAL_COL].sort_values().index
        trace_df = trace_df.loc[order]
        if len(trace_df) >= min_length:
            traces_lst.append(Trace.from_raw_trace_data(trace_df, agg_window))

def build_all_traces_from_files(trace_files, min_length, agg_window):
    """Builds a list of traces from `trace_files`.

    The traces are built using parallel processing.

    Parameters
    ----------
    trace_files: list
        A list of strings representing the files from which the traces
        are built.
    min_length: int
        An integer representing the minimum number of time points needed
        for a trace to be considered.
    agg_window: int
        An integer representing the aggregation window for the trace.

    Returns
    -------
    list
        A list of `Traces` obtained from the trace files.

    """
    results = mp.Manager().list()
    procs = []
    cores, traces_per_core = get_cores_and_traces_per_core(len(trace_files))
    for core_num in range(cores):
        core_trace_files = get_traces_for_core(
            trace_files, traces_per_core, core_num)
        procs.append(
            mp.Process(target=build_traces_from_files,
                       args=(core_trace_files, results,
                             min_length, agg_window)))
    initialize_and_join_processes(procs)
    return list(results)


def get_multivariate_model_results(model_params, train_df, train_preds,
                                   test_df, test_preds, model_vars):
    """Calculates the model results for `model_vars`.

    A separate `ModelResults` object is built for each variable in
    `model_vars` based on the actual and predicted values in the training
    and test sets.

    Parameters
    ----------
    model_params: tuple
        A tuple specifying the parameters of the trained model.
    train_df: pd.DataFrame
        A pandas DataFrame containing the target data for the training set.
    train_preds: pd.DataFrame
        A pandas DataFrame containing the predictions for the training set.
    test_df: pd.DataFrame
        A pandas DataFrame containing the target data for the testing set.
    test_preds: pd.DataFrame
        A pandas DataFrame containing the predictions for the testing set.
    model_vars: list
        A list of strings with the names of the variables being modeled.

    Returns
    -------
    dict
        A dictionary of model results. The keys are the variable names
        specified in `model_vars` and the corresponding value is a
        `ModelResults` object corresponding to the results for that variable.

    """
    results_dict = {}
    for model_var in model_vars:
        results_dict[model_var] = ModelResults(model_params,
            train_df[model_var].values, train_preds[model_var].values,
            test_df[model_var].values, test_preds[model_var].values)
    return results_dict
