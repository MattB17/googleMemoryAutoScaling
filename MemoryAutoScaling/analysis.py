"""A set of helper functions used in the analysis process.

"""
import sys
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import product
from MemoryAutoScaling.DataHandling import TraceHandler


def get_granger_pvalues_at_lag(granger_dict, lag):
    """Retrieves the pvalues from `granger_dict` at `lag`.

    Parameters
    ----------
    granger_dict: dict
        A dictionarity containing the results of a Granger test for causality.
    lag: int
        An integer representing the lag used for the causality test.

    Returns
    -------
    list
        A list of pvalues from the granger causality test recorded in
        `granger_dict` at the lag `lag`.

    """
    granger_dict = granger_dict[lag][0]
    return [granger_dict['ssr_ftest'][1],
            granger_dict['ssr_chi2test'][1],
            granger_dict['lrtest'][1],
            granger_dict['params_ftest'][1]]


def get_granger_col_names_for_lag(lag):
    """The set of granger column names for `lag`.

    Parameters
    ----------
    lag: int
        An integer representing the time lag of interest.

    Returns
    -------
    list
        A list of strings representing the column names for the granger
        causality test at a lag of `lag`.

    """
    test_names = ['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest']
    return ["{0}_{1}".format(test_name, lag) for test_name in test_names]


def get_all_granger_col_names(causal_cols, causal_lags):
    """Gets all granger column names for `causal_cols` and `causal_lags`.

    That is there is a column for each combination of `causal_cols`,
    `causal_lags`, and each statistical test.

    Parameters
    ----------
    causal_cols: list
        A list of strings representing the columns for which a test was
        carried out to determine if the given column is causally related
        to the target variable.
    causal_lags: list
        A list of integers representing the lags tested for causality.

    Returns
    -------
    list
        A list of strings representing all granger column names for
        `causal_cols` and `causal_lags`.

    """
    causal_lst = [get_granger_col_names_for_lag(lag) for lag in causal_lags]
    causal_lst = [col_name for lag_list in causal_lst
                  for col_name in lag_list]
    return ["causal_{0}_{1}".format(causal_tup[0], causal_tup[1])
            for causal_tup in product(causal_cols, causal_lst)]


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


def build_models_from_params_list(time_series_model, params_lst):
    """Builds `time_series_model` objects from the params in `params_lst`.

    A separate `time_series_model` object is built for each set of params
    in `params_lst`.

    Parameters
    ----------
    time_series_model: TimeSeriesModel.class
        A reference to a `TimeSeriesModel` class representing the models being
        created.
    params_lst: list
        A list of dictionaries in which each dictionary represents the set of
        named parameters used to initialize a `time_series_model`.

    Returns
    -------
    list
        A list of `time_series_model` objects instantiated from the parameters
        in `params_lst`.

    """
    return [time_series_model(**params) for params in params_lst]


def get_model_stats_for_trace(data_trace, models):
    """Gets statistics from `models` for `data_trace`.

    For each model in `models`, the model is fit to `data_trace` and
    the mean squared error on the test set is computed.

    Parameters
    ----------
    data_trace: Trace
        A `Trace` representing the data trace from which the statistics
        will be calculated.
    models: list
        A list of `TimeSeriesModel` objects that will be fit to `data_trace`.

    Returns
    -------
    list
        A list containing the ID of `data_trace` followed by the mean squared
        error on the training and test set, respectively, for each model in
        `models`.

    """
    trace_stats = [data_trace.get_trace_id()]
    for model in models:
        train_mse, test_mse = model.calculate_train_and_test_mse(
            data_trace.get_maximum_memory_time_series())
        trace_stats.extend([train_mse, test_mse])
    return trace_stats


def read_modelling_input_params():
    """Reads the input parameters for modelling from standard input.

    The input paramters consist of the input directory containing the
    trace files, the output directory for modelling results, a string
    identifying files to read from the input directory, an integer
    representing the minimum length for a trace to be modelled, and a float
    representing the proportion of data in the training set.

    Returns
    -------
    dict
        A dictionary containing the input parameters.

    """
    return {"input_dir": sys.argv[1],
            "output_dir": sys.argv[2],
            "file_id": sys.argv[3],
            "min_trace_length": int(sys.argv[4]),
            "train_prop": float(sys.argv[5])}


def get_model_build_input_params():
    """Gets the set of parameters to build models for the traces.

    Reads the input arguments from the system input and uses these to setup
    the input parameters for model building. These input parameters are
    the collection of traces to be modeled, an output directory for model
    results, and the proportion of records in the training set.

    Returns
    -------
    list, str, float
        A list of `Trace` objects representing the traces to be modeled, a
        string representing the output directory for model results, and a
        float in the range [0, 1] representing the proportion of observations
        in the training set.

    """
    params = read_modelling_input_params()
    trace_handler = TraceHandler(
        params['input_dir'], params['file_id'], params['min_trace_length'])
    traces = trace_handler.run_processing_pipeline()
    return traces, params['output_dir'], params['train_prop']


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
    list
        A list of lists in which each list contains the trace ID and model
        results for that trace, for each trace in `traces`.

    """
    results = mp.Manager().list()
    procs = []
    cores, traces_per_core = get_cores_and_traces_per_core(len(traces))
    for core_num in range(cores):
        core_traces = get_traces_for_core(traces, traces_per_core, core_num)
        procs.append(mp.Process(target=model_func,
                                args=(core_traces, results, train_prop)))
    initialize_and_join_processes(procs)
    return list(results)


def output_model_results(results, col_list, output_dir, file_name):
    """Outputs `results` to `file_name` in `output_dir`.

    The model results are converted to a pandas DataFrame and written to
    a csv file in `output_dir` named `file_name`.

    Parameters
    ----------
    results: list
        A list of lists containing the model results. Each list corresponds
        to the results for a particular trace.
    col_list: list
        A list of strings representing the column names for the data.
    output_dir: str
        A string representing the directory to which the results are written.
    file_name: str
        A string representing the name of the file to which the results are
        written.

    Returns
    -------
    None

    """
    results_df = pd.DataFrame(results)
    results_df.columns = col_list
    output_path = os.path.join(output_dir, "{}.csv".format(file_name))
    results_df.to_csv(output_path, sep=",", index=False)


def get_col_list_for_params(params, model_name):
    """Gets a list of columns based on `params`.

    Parameters
    ----------
    params: list
        A list of model parameters used to generate the column names.
    model_name: str
        A string representing the name of the model.

    Returns
    -------
    list
        A list consisting of column names generated from `params`.

    """
    return ["{0}_mse_{1}_{2}".format(mse_name, model_name, param)
            for param, mse_name in product(params, ["train", "test"])]


def model_traces_and_evaluate(model, model_params, traces, results_lst):
    """Fits `model` to `traces` and evaluates the performance.

    A separate `model` is built for each parameter set in `model_params` and
    fit to the `Trace` objects in `traces`. The models are then evaluated, and
    the results are saved in `results_lst`.

    Parameters
    ----------
    model: TimeSeriesModel.class
        A `TimeSeriesModel` class reference specifying the type of model to
        be built.
    model_params: list
        A list of dictionaries where each dictionary corresponds to the set
        of parameters to build a model of type `model`.
    traces: list
        A list of `Trace` objects specifying the traces to be modelled.
    results_lst: mp.Manager.list
        A multiprocessor list representing the list to which model results are
        saved.

    Returns
    -------
    None

    """
    models = build_models_from_params_list(model, model_params)
    for trace in traces:
        results_lst.append(get_model_stats_for_trace(trace, models))

def insert_model_results_at_index(results_lst, model_params,
                                  train_mse, test_mse, idx):
    """Inserts model results into `results_lst` based on `idx`.

    `model_params`, `train_mse`, and `test_mse` are inserted into
    `results_lst` at positions `3 * idx + 1`, `3 * idx + 2` and
    `3 * idx + 3`, respectively.

    Parameters
    ----------
    results_lst: list
        A list of results for which the model results are inserted.
    model_params: tuple
        A tuple containing model parameters for the results being inserted.
    train_mse: float
        A float representing the training MSE for the model.
    test_mse: float
        A float representing the test MSE for the model.
    idx: int
        An integer used to mark where the model results will be inserted.

    Returns
    -------
    list
        The list obtained from `results_lst` after inserting the model results.

    """
    results_lst.insert((3 * idx) + 1, model_params)
    results_lst.insert((3 * idx) + 2, train_mse)
    results_lst.insert((3 * idx) + 3, test_mse)
    return results_lst

def extend_model_results_up_to_cutoff(results_lst, model_params, train_mse,
                                      test_mse, cutoff):
    """Extends `results_lst` with the model results up to `cutoff`.

    If `results_lst` has less than `cutoff` elements then it is extended with
    the model results given by `model_params`, `train_mse` and `test_mse`.

    Parameters
    ----------
    results_lst: list
        The list containing model results that is being extended.
    model_params: tuple
        A tuple containing the model params for the model.
    train_mse: float
        A float representing the training MSE for the model.
    test_mse: float
        A float representing the testing MSE for the model.
    cutoff: int
        An integer used to decide if `results_lst` should be extended. If the
        length of `results_lst` is below `cutoff` then it is extended with the
        model results.

    Returns
    -------
    list
        The list obtained `results_lst` after possibly extending with the
        model results.

    """
    if len(results_lst) < cutoff:
        results_lst.extend([model_params, train_mse, test_mse])
    return results_lst


def update_with_model_results(results_lst, model_params, train_mse,
                              test_mse, cutoff):
    """Updates `results_lst` with the model results.

    If `results_lst` has fewer than `cutoff` entries or `test_mse` is lower
    than the test MSE of a model already contained in `results_lst`, then
    the model results are inserted into `results_lst`.

    Parameters
    ----------
    results_lst: list
        The list containing model results that is being updated.
    model_params: tuple
        A tuple containing the model params for the model.
    train_mse: float
        A float representing the training MSE for the model.
    test_mse: float
        A float representing the testing MSE for the model.
    cutoff: int
        An integer used to decide if `results_lst` should be extended. If the
        length of `results_lst` is below `cutoff` then it is extended with the
        model results.

    Returns
    -------
    list
        The list obtained `results_lst` after possibly extending or inserting
        the model results.

    """
    model_count = (len(results_lst) - 1) // 3
    for idx in range(model_count):
        if test_mse < results_lst[3 * (idx + 1)]:
            return insert_model_results_at_index(
                results_lst, model_params, train_mse, test_mse, idx)
    return extend_model_results_up_to_cutoff(
        results_lst, model_params, train_mse, test_mse, cutoff)
