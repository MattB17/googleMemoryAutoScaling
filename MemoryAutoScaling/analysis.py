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
        model_stats = list(model.run_model_pipeline_for_trace(data_trace))
        trace_stats.extend(model_stats)
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


def get_col_list_for_params(params, model_name, model_cols):
    """Gets a list of columns based on `params` and `model_cols`.

    Parameters
    ----------
    params: list
        A list of model parameters used to generate the column names.
    model_name: str
        A string representing the name of the model.
    model_cols: list
        A list of column names for the models.

    Returns
    -------
    list
        A list consisting of column names generated from `params` and
        `model_cols`.

    """
    return ["{0}_{1}_{2}".format(model_col, model_name, param)
            for param, model_col in product(params, model_cols)]


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

def insert_model_results_at_index(results_lst, model_results, idx):
    """Inserts model results into `results_lst` based on `idx`.

    `model_results` are inserted into `results_lst` starting at position
    `n * idx + 1` where `n` is the length of `model_results`.

    Parameters
    ----------
    results_lst: list
        A list of results for which the model results are inserted.
    model_results: list
        A list containing the model results.
    idx: int
        An integer used to mark where the model results will be inserted.

    Returns
    -------
    list
        The list obtained from `results_lst` after inserting the model results.

    """
    n = len(model_results)
    for i in range(n):
        results_lst.insert((n * idx) + i + 1, model_results[i])
    return results_lst

def extend_model_results_up_to_cutoff(results_lst, model_results, cutoff):
    """Extends `results_lst` with the model results up to `cutoff`.

    If `results_lst` has less than `cutoff` elements then it is extended with
    the model results given by `model_results`.

    Parameters
    ----------
    results_lst: list
        The list containing model results that is being extended.
    model_results: list
        A list containing the model results.
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
        results_lst.extend(model_results)
    return results_lst


def update_with_model_results(results_lst, model_results, cutoff):
    """Updates `results_lst` with the model results.

    If `results_lst` has fewer than `cutoff` entries or `test_mse` is lower
    than the test MSE of a model already contained in `results_lst`, then
    the model results are inserted into `results_lst`.

    Parameters
    ----------
    results_lst: list
        The list containing model results that is being updated.
    model_results: list
        A list containing model results where the third element is the mean
        squared error for the test set.
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
    model_count = (len(results_lst) - 1) // len(model_results)
    for idx in range(model_count):
        if model_results[2] < results_lst[len(model_results) * (idx + 1)]:
            return insert_model_results_at_index(
                results_lst, model_results, idx)
    return extend_model_results_up_to_cutoff(
        results_lst, model_results, cutoff)


def truncate_list(lst, cutoff):
    """Truncates `lst` if it is longer than `cutoff`.

    The size of `lst` is reduced to `cutoff` if it is longer than `cutoff`.
    Otherwise, it is unchanged.

    Parameters
    ----------
    lst: list
        The list being truncated.
    cutoff: int
        An integer representing the maximum length for `lst`.

    Returns
    -------
    list
        The list resulting from truncating `lst` with `cutoff`.

    """
    if len(lst) > cutoff:
        return lst[:cutoff]
    return lst


def handle_stats_for_model(trace, model, trace_results, cutoff):
    """Handles the stats for `model` built on `trace`.

    `trace` is modeled using `model` and the results are added to
    `trace_results` if it has fewer than `cutoff` entries or if the results
    for the model are better than the results of at least one model contained
    in `trace_results`.

    Parameters
    ----------
    trace: Trace
        The `Trace` object being modeled.
    model: TimeSeriesModel
        A `TimeSeriesModel` to be fit to `trace`.
    trace_results: list
        A list containing results for modelling done on the trace.
    cutoff: int
        An integer representing the maximum length of `trace_results`.

    Returns
    -------
    list
        A list obtained from `trace_results` after handling the results of
        modeling `trace` with `model`.

    """
    model_results = list(model.run_model_pipeline_for_trace(trace))
    trace_results = update_with_model_results(
        trace_results, [model.get_params()] + model_results, cutoff)
    return truncate_list(trace_results, cutoff)


def update_model_stats_for_trace(trace, model, model_results, cutoff):
    """Updates the stats for `model` built on `trace`.

    `trace` is modeled using `model` and the results are added to
    `trace_results` if it has fewer than `cutoff` entries or if the results
    for the model are better than the results of at least one model contained
    in `trace_results`.

    Parameters
    ----------
    model: TimeSeriesModel
        A `TimeSeriesModel` to be fit to `trace`.
    trace: Trace
        The `Trace` object being modeled.
    trace_results: list
        A list containing results for modelling done on the trace.
    cutoff: int
        An integer representing the maximum length of `trace_results`.

    Returns
    -------
    list
        A list obtained from `trace_results` after handling the results of
        modeling `trace` with `model`.

    """
    try:
        return handle_stats_for_model(trace, model, model_results, cutoff)
    except:
        return model_results


def pad_list(lst, pad_val, pad_len):
    """Pads `lst` with `pad_val` to length `pad_len`.

    If the length of `lst` is already greater or equal to `pad_len` then
    `lst` is not modified.

    Parameters
    ----------
    lst: list
        The list being padded.
    pad_val: float
        The value used to pad `lst`.
    pad_len: int
        An integer representing the length to which `lst` should be padded.

    Returns
    -------
    list
        The list obtained from `lst` after padding with `pad_val` up to length
        `pad_len`.

    """
    if len(lst) < pad_len:
        lst += [pad_val for _ in range(pad_len - len(lst))]
    return lst


def get_best_models_for_trace(trace, models, models_count):
    """Gets statistics for the best models in `models` for `trace`.

    The best models in `models` are the `models_count` models with the lowest
    test MSE when built on `trace`.

    Parameters
    ----------
    trace: Trace
        The `Trace` object being modeled.
    models: list
        A list of `TimeSeriesModel` objects representing the models being fit
        to `trace` and from which the best models are chosen.
    models_count: int
        An integer representing the number of models to include in the results.

    Returns
    -------
    list
        A list consisting of the ID of `trace` followed by the results for the
        `models_count` best models from `models` fit to `trace`.

    """
    best_results = [trace.get_trace_id()]
    cutoff = (5 * models_count) + 1
    for model in models:
        best_results = update_model_stats_for_trace(
            trace, model, best_results, cutoff)
    return pad_list(best_results, np.nan, cutoff)

def get_best_model_results_for_traces(model, model_params, traces,
                                      result_lst, models_count):
    """Gets the `models_count` best model results for the traces of `traces`.

    For each trace in `traces` a `model` object is built for each set of model
    parameters in `model_params` and the results of the best `models_count`
    models are saved in `result_lst`. If fewer than `models_count` models are
    fitted to the trace, then the result is padded with `np.nan`.

    Parameters
    ----------
    model: TimeSeriesModel.class
        A `TimeSeriesModel` class specifying the model to be built.
    model_params: list
        A list of dictionaries specifying the model parameters for each model
        to be built.
    traces: list
        A list of `Trace` objects specifying the traces to which models are
        fit.
    result_lst: list
        The list to which the model results for each trace are saved.
    models_count: int
        An integer representing the number of model results to save. The
        results for the `models_count` best models for each trace will be
        saved.

    Returns
    -------
    None

    """
    models = build_models_from_params_list(model, model_params)
    for trace in traces:
        result_lst.append(
            get_best_models_for_trace(trace, models, models_count))

def run_models_for_all_traces(modeling_func, model_params, model_name):
    """Models all traces using `modeling_func` with `model_name`.

    A series of models are run on each trace, where the models are specified
    by the parameters in `model_params`. `modeling_func` is used to run the
    models on a batch of traces and this function is parallelized to cover all
    traces.

    Parameters
    ----------
    modeling_func: function
        The function used to perform the modelling on a batch of traces. The
        function takes three arguments: a list of `Trace` objects on which the
        models are run, a list to which results are saved, and a float
        specifying the proportion of data in the training set.
    model_params: list
        A list containing the model parameters for the models.
    model_name: str
        A string specifying the name of the model.

    Returns
    -------
    None

    Side Effect
    -----------
    The results of the modelling procedure is saved to a csv file containing
    one row per trace. The file is output to a file named `<name>_results.csv`
    where `<name>` is `model_name`.

    """
    traces, output_dir, train_prop = get_model_build_input_params()
    results = perform_trace_modelling(traces, modeling_func, train_prop)
    cols = get_col_list_for_params(
        model_params, model_name,
        ["train_mse", "test_mse", "num_under_preds", "max_under_pred"])
    output_model_results(
        results, ["id"] + cols, output_dir, "{}_results".format(model_name))

def run_best_models_for_all_traces(modeling_func, models_count, model_name):
    """Models all traces using `modeling_func` with `model_name`.

    A series of models are run on each trace, and the best `models_count` of
    these models for each trace are recorded. `modeling_func` is used to run
    the models on a batch of traces and this function is parallelized to cover
    all traces. The best models are then retrieved.

    Parameters
    ----------
    modeling_func: function
        The function used to perform the modelling on a batch of traces. The
        function takes three arguments: a list of `Trace` objects on which the
        models are run, a list to which results are saved, and a float
        specifying the proportion of data in the training set.
    models_count: int
        An integer representing the number of models to record. The results
        for the `models_count` best models for each trace are recorded.
    model_name: str
        A string specifying the name of the model.

    Returns
    -------
    None

    Side Effect
    -----------
    The results of the modelling procedure is saved to a csv file containing
    one row per trace. The file is output to a file named `<name>_results.csv`
    where `<name>` is `model_name`.

    """
    traces, output_dir, train_prop = get_model_build_input_params()
    model_cols = ["params", "train_mse", "test_mse",
                  "num_under_preds", "max_under_pred"]
    results = perform_trace_modelling(traces, modeling_func, train_prop)
    cols = get_col_list_for_params(
        range(1, models_count + 1), model_name, model_cols)
    output_model_results(
        results, ["id"] + cols, output_dir, "{}_results".format(model_name))
