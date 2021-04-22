"""A set of helper functions used in the analysis process.

"""
import sys
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import product
from MemoryAutoScaling.Analysis import ModelResults
from MemoryAutoScaling import parallel, specs, utils
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


def read_univariate_modelling_input_params():
    """Reads the input parameters for modelling from standard input.

    The input paramters consist of the input directory containing the
    trace files, the output directory for modelling results, a string
    identifying files to read from the input directory, an integer
    representing the minimum length for a trace to be modelled, a float
    representing the proportion of data in the training set, an integer
    representing for how many successive periods to aggregate data, and
    a boolean value indicating the target variable.

    Returns
    -------
    dict
        A dictionary containing the input parameters.

    """
    return {"input_dir": sys.argv[1],
            "output_dir": sys.argv[2],
            "file_id": sys.argv[3],
            "min_trace_length": int(sys.argv[4]),
            "train_prop": float(sys.argv[5]),
            "aggregation_window": int(sys.argv[6]),
            "max_mem": (sys.argv[7].lower() == "true")}


def read_multivariate_modelling_input_params():
    """Reads the input parameters for modelling from standard input.

    The input paramters consist of the input directory containing the
    trace files, the output directory for modelling results, a string
    identifying files to read from the input directory, an integer
    representing the minimum length for a trace to be modelled, a float
    representing the proportion of data in the training set, and an integer
    representing for how many successive periods to aggregate data.

    Returns
    -------
    dict
        A dictionary containing the input parameters.

    """
    return {"input_dir": sys.argv[1],
            "output_dir": sys.argv[2],
            "file_id": sys.argv[3],
            "min_trace_length": int(sys.argv[4]),
            "train_prop": float(sys.argv[5]),
            "aggregation_window": int(sys.argv[6])}


def get_traces_from_input_params(input_params):
    """Gets the collection of traces based on `input_params`.

    Parameters
    ----------
    input_params: dict
        A dictionary containing the input parameters used to construct the
        traces. The dictionary has keys 'input_dir', 'file_id',
        'min_trace_length', and 'aggregation_window'.

    Returns
    -------
    list
        A list of `Trace` objects constructed based on `input_params`.

    """
    trace_handler = TraceHandler(
        input_params['input_dir'], input_params['file_id'],
        input_params['min_trace_length'], input_params['aggregation_window'])
    return trace_handler.run_processing_pipeline()


def get_univariate_model_build_input_params():
    """Gets the set of parameters to build univariate models for the traces.

    Reads the input arguments from the system input and uses these to setup
    the input parameters for model building. These input parameters are
    the collection of traces to be modeled, an output directory for model
    results, and the proportion of records in the training set.

    Returns
    -------
    dict
        A dictionary containing a list of traces, the output directory for
        model results, a float representing the proportion of records in the
        training set, and a boolean indicating which target variable to use
        for modeling.

    """
    params = read_univariate_modelling_input_params()
    return {'traces': get_traces_from_input_params(params),
            'output_dir': params['output_dir'],
            'train_prop': params['train_prop'],
            'max_mem': params['max_mem']}


def get_multivariate_model_build_input_params():
    """Gets the set of parameters to build multivariate models for the traces.

    Reads the input arguments from the system input and uses these to setup
    the input parameters for model building. These input parameters are
    the collection of traces to be modeled, an output directory for model
    results, and the proportion of records in the training set.

    Returns
    -------
    dict
        A dictionary containing the list of traces to be modeled, the output
        directory for modeling results, and a float representing the
        proportion of records to use in the training set.

    """
    params = read_multivariate_modelling_input_params()
    return {"traces": get_traces_from_input_params(params),
            "output_dir": params['output_dir'],
            "train_prop": params['train_prop']}


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
    if len(params) > 1:
        return ["{0}_{1}_{2}".format(model_col, model_name, param)
                for param, model_col in product(params, model_cols)]
    return ["{0}_{1}".format(model_col, model_name)
            for model_col in model_cols]


def update_with_new_model_results(model_results, new_model_results):
    """Updates `model_results` with `new_model_results`.

    `new_model_results` is inserted into its appropriate place in the sorted
    list `model_results`, based on the ranking metric defined on
    `ModelResults` objects.

    Parameters
    ----------
    model_results: list
        The list of `ModelResults` that is being updated.
    new_model_results: ModelResults
        A `ModelResults` object representing the most recent model results

    Returns
    -------
    list
        The list obtained `model_results` inserting `new_model_results`.

    """
    for idx in range(len(model_results)):
        old_model_results = model_results[idx]
        if new_model_results.is_better(old_model_results):
            model_results.insert(idx, new_model_results)
            return model_results
    model_results.append(new_model_results)
    return model_results

def insert_multivariate_results(model_results_dict, new_results_dict, idx):
    """Inserts `new_results_dict` into `model_results_dict` at `idx`.

    Parameters
    ----------
    model_results_dict: dict
        A dictionary containing lists of `ModelResults`. Each key is a string
        representing a modeled variable and the corresponding value is a list
        of `ModelResults` objects for that variable.
    new_results_dict: dict
        A dictionary of `ModelResults`. Each key is a string representing a
        modeled variable and the corresponding value is an associated
        `ModelResults` object.
    idx: int
        An integer representing the index to which the `ModelResults` of
        `new_results_dict` will be inserted into the lists of
        `model_results_dict`.

    Returns
    -------
    dict
        The dictionary obtained from `model_results_dict` after inserting
        `new_results_dict` at `idx`.

    """
    for model_var in model_results_dict.keys():
        model_results_dict[model_var].insert(idx, new_results_dict[model_var])
    return model_results_dict

def append_multivariate_results(model_results_dict, new_results_dict):
    """Appends `new_results_dict` into `model_results_dict`.

    Parameters
    ----------
    model_results_dict: dict
        A dictionary containing lists of `ModelResults`. Each key is a string
        representing a modeled variable and the corresponding value is a list
        of `ModelResults` objects for that variable.
    new_results_dict: dict
        A dictionary of `ModelResults`. Each key is a string representing a
        modeled variable and the corresponding value is an associated
        `ModelResults` object.

    Returns
    -------
    dict
        The dictionary obtained from `model_results_dict` after inserting
        `new_results_dict` at `idx`.

    """
    for model_var in model_results_dict.keys():
        model_results_dict[model_var].append(new_results_dict[model_var])
    return model_results_dict

def update_with_new_multivariate_model_results(model_results_dict,
                                               new_results_dict):
    """Updates `model_results_dict` with the new model results.

    The `ModelResults` in `new_results_dict` are inserted into the appropriate
    place in `model_results_dict`, based on the ranking metric defined on
    `ModelResults` objects.

    Parameters
    ----------
    model_results_dict: dict
        The dictionary containing model results that is being updated.
    new_results_dict: dict
        A dictionary of new model results. The keys are strings representing
        the target variables being modeled. The corresponding value is a
        `ModelResults` object for that variable.

    Returns
    -------
    dict
        The dictionary obtained from `model_results_dict` after possibly
        extending or inserting the `ModelResults` from `new_results_dict`.

    """
    new_model_results = new_results_dict[specs.MAX_MEM_TS]
    for idx in range(len(model_results_dict[specs.MAX_MEM_TS])):
        old_model_results =  model_results_dict[specs.MAX_MEM_TS][idx]
        if new_model_results.is_better(old_model_results):
            return insert_multivariate_results(
                model_results_dict, new_results_dict, idx)
    return append_multivariate_results(model_results_dict, new_results_dict)

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


def truncate_dict(raw_dict, cutoff):
    """Truncates each list of `raw_dict` if it is longer than `cutoff`.

    The size of each list in `raw_dict` is reduced to `cutoff` if it is longer
    than `cutoff`. Otherwise, it is unchanged.

    Parameters
    ----------
    raw_dict: dict
        The dictionary being truncated.
    cutoff: int
        An integer representing the maximum length for the lists of `raw_dict`.

    Returns
    -------
    dict
        The dictionary resulting from truncating each list of `raw_dict` to a
        length of `cutoff`.

    """
    for model_var in raw_dict.keys():
        raw_dict[model_var] = truncate_list(raw_dict[model_var], cutoff)
    return raw_dict


def handle_results_for_model(trace, model, model_results, cutoff):
    """Handles the model results for `model` built on `trace`.

    `trace` is modeled using `model` and the results are added to
    `model_results` if it has fewer than `cutoff` entries or if the results
    for the model are better than the results of at least one model contained
    in `model_results`.

    Parameters
    ----------
    trace: Trace
        The `Trace` object being modeled.
    model: TraceModel
        A `TraceModel` to be fit to `trace`.
    model_results: list
        A list of `ModelResults` objects for the results of models built on
        the trace.
    cutoff: int
        An integer representing the maximum length of `model_results`.

    Returns
    -------
    list
        A list obtained from `model_results` after handling the results of
        modeling `trace` with `model`.

    """
    new_model_results = model.run_model_pipeline_for_trace(trace)
    model_results = update_with_new_model_results(
        model_results, new_model_results)
    return truncate_list(model_results, cutoff)


def initialize_multivariate_results(model_results_dict):
    """Initializes the multivariate results from `model_results_dict`.

    Parameters
    ----------
    model_results_dict: dict
        A dictionary of `ModelResults`. The keys are strings representing
        one of the variables being modelled. The corresponding value is a
        `ModelResults` object associated with that variable.

    Returns
    -------
    dict
        A dictionary of model results in which the keys are strings
        representing the model variables specified in `model_results_dict`
        and the corresponding value is a 1-element list containing the
        associate `ModelResults` object from `model_results_dict`.

    """
    return {model_var: [model_results] for model_var, model_results
            in model_results_dict.items()}

def handle_results_for_multivariate_model(trace, model, model_results, cutoff):
    """Handles the model results for `model` built on `trace`.

    `trace` is modeled using `model` and the results are added to
    `model_results` if it has fewer than `cutoff` entries or if the results
    for the model are better than the results of at least one model contained
    in `model_results`.

    Parameters
    ----------
    trace: Trace
        The `Trace` object being modeled.
    model: TraceModel
        A `TraceModel` to be fit to `trace`.
    model_results: dict
        A dictionary containing results for modelling done on the trace.
    cutoff: int
        An integer representing the maximum length of `results`.

    Returns
    -------
    dict
        A dictionary obtained from `model_results` after handling the results
        of modeling `trace` with `model`.

    """
    new_model_results_dict = model.run_model_pipeline_for_trace(trace)
    if model_results == {}:
        model_results = initialize_multivariate_results(new_model_results_dict)
    else:
        model_results = update_with_new_multivariate_model_results(
            model_results, new_model_results_dict)
    return truncate_dict(model_results, cutoff)


def update_model_results_for_trace(trace, model, model_results, cutoff):
    """Updates `model_results` for `model` built on `trace`.

    `trace` is modeled using `model` and the results are added to
    `model_results` if it has fewer than `cutoff` entries or if the results
    for the model are better than the results of at least one model contained
    in `model_results`.

    Parameters
    ----------
    model: TraceModel
        A `TraceModel` to be fit to `trace`.
    trace: Trace
        The `Trace` object being modeled.
    trace_results: list
        A list containing results for modelling done on the trace.
    cutoff: int
        An integer representing the maximum length of `trace_results`.

    Returns
    -------
    list
        A list obtained from `model_results` after handling the results of
        modeling `trace` with `model`.

    """
    try:
        return handle_results_for_model(trace, model, model_results, cutoff)
    except:
        return model_results

def update_multivariate_model_results_for_trace(trace, model,
                                                model_results, cutoff):
    """Updates `model_results` for `model` built on `trace`.

    `trace` is modeled using `model` and the results are added to
    `model_results` if it has fewer than `cutoff` entries or if the results
    for the model are better than the results of at least one model contained
    in `model_results`.

    Parameters
    ----------
    model: TraceModel
        A `TraceModel` to be fit to `trace`.
    trace: Trace
        The `Trace` object being modeled.
    model_results: dict
        A dictionary containing results for modelling done on the trace.
    cutoff: int
        An integer representing the maximum length of `results`.

    Returns
    -------
    dict
        The dictionary obtained from `model_results` after handling the
        results of modeling `trace` with `model`.

    """
    try:
        return handle_results_for_multivariate_model(
            trace, model, model_results, cutoff)
    except:
        return model_results


def pad_model_results(model_results_lst, pad_len):
    """Pads `model_results_lst` with null `ModelResults` to length `pad_len`.

    Parameters
    ----------
    model_results_lst: list
        The list of `ModelResults` objects being padded.
    pad_len: int
        An integer representing the length to which `model_results_lst`
        should be padded.

    Returns
    -------
    list
        The list obtained from `model_results_lst` after padding with null
        `ModelResults` objects up to length `pad_len`.

    """
    if len(model_results_lst) < pad_len:
        model_results_lst += [ModelResults.build_null_model_results() for _
                              in range(pad_len - len(model_results_lst))]
    return model_results_lst


def pad_model_results_dict(model_results_dict, pad_len):
    """Pads `models_results_dict` with null `ModelResults` to `pad_len`.

    Each list of `model_results_dict` is padded with null `ModelResults`
    objects up to `pad_len`.

    Parameters
    ----------
    model_results_dict: dict
        A dictionary of `ModelResults`. They keys are strings representing the
        names of the target variables and the corresponding value is a list of
        `ModelResults` for that variable which will be padded to `pad_len`.
    pad_len: int
        An integer representing the length to which each list of
        `model_results_dict` will be padded.

    Returns
    -------
    dict
        The dictionary obtained from `model_results_dict` after padding each
        list with null `ModelResults` up to `pad_len`.

    """
    for model_var in model_results_dict.keys():
        model_results_dict[model_var] = pad_model_results(
            model_results_dict[model_var], pad_len)
    return model_results_dict


def compute_best_results_for_trace(trace, models, models_count):
    """Gets `ModelResults` for the best models of `models` for `trace`.

    The best models of `models` are the `models_count` models with the lowest
    test MASE when built on `trace`.

    Parameters
    ----------
    trace: Trace
        The `Trace` object being modeled.
    models: list
        A list of `TraceModel` objects representing the models being fit
        to `trace` and from which the best models are chosen.
    models_count: int
        An integer representing the number of models to include in the results.

    Returns
    -------
    list
        A list consisting of the ID of `trace` followed by the results for the
        `models_count` best models from `models` fit to `trace`.

    """
    best_results = []
    for model in models:
        best_results = update_model_results_for_trace(
            trace, model, best_results, models_count)
    return pad_model_results(best_results, models_count)

def compute_best_multivariate_results_for_trace(trace, models, models_count):
    """Gets `ModelResults` for the best models of `models` for `trace`.

    The best multivariate models in `models` are the `models_count` models
    with the lowest test MASE when built on `trace`.

    Parameters
    ----------
    trace: Trace
        The `Trace` object being modeled.
    models: list
        A list of `TraceModel` objects representing the models being fit
        to `trace` and from which the best models are chosen.
    models_count: int
        An integer representing the number of models to include in the results.

    Returns
    -------
    dict
        A dictionary of model results. The keys are strings representing the
        names of the variables being modeled. The corresponding value is a
        list of `ModelResults` objects representing model results for `trace`
        on the associated variable.

    """
    best_results_dict = {}
    for model in models:
        best_results_dict = update_multivariate_model_results_for_trace(
            trace, model, best_results_dict, models_count)
    return pad_model_results_dict(best_results_dict, models_count)

def get_best_model_results_for_traces(model, model_params, traces,
                                      result_dict, models_count, verbose=True):
    """Gets the `models_count` best model results for the traces of `traces`.

    For each trace in `traces` a `model` object is built for each set of model
    parameters in `model_params` and the results of the best `models_count`
    models are saved in `result_dict`.

    Parameters
    ----------
    model: TraceModel.class
        A `TraceModel` class specifying the model to be built.
    model_params: list
        A list of dictionaries specifying the model parameters for each model
        to be built.
    traces: list
        A list of `Trace` objects specifying the traces to which models are
        fit.
    result_dict: dict
        The dictionary to which the model results for each trace are saved.
    models_count: int
        An integer representing the number of model results to save. The
        results for the `models_count` best models for each trace will be
        saved.
    verbose: bool
        A boolean indicating whether to run in verbose mode. The default
        value is True, in which case progress is printed to the screen.

    Returns
    -------
    None

    """
    models = build_models_from_params_list(model, model_params)
    trace_count = len(traces)
    for idx in range(trace_count):
        model_results = compute_best_results_for_trace(
            traces[idx], models, models_count)
        result_dict[traces[idx].get_trace_id()] = model_results
        log_modeling_progress(idx, trace_count, verbose)

def get_best_multivariate_model_results_for_traces(
    model, model_params, traces, result_dict, models_count, verbose=True):
    """Gets the `models_count` best model results for the traces of `traces`.

    For each trace in `traces` a `model` object is built for each set of model
    parameters in `model_params` and the results of the best `models_count`
    models are saved in `result_dict`.

    Parameters
    ----------
    model: TraceModel.class
        A `TraceModel` class specifying the model to be built.
    model_params: list
        A list of dictionaries specifying the model parameters for each model
        to be built.
    traces: list
        A list of `Trace` objects specifying the traces to which models are
        fit.
    result_dict: dict
        The dictionary to which the model results for each trace are saved.
    models_count: int
        An integer representing the number of model results to save. The
        results for the `models_count` best models for each trace will be
        saved.
    verbose: bool
        A boolean indicating whether to run in verbose mode. The default
        value is True, in which case progress is printed to the screen.

    Returns
    -------
    None

    """
    models = build_models_from_params_list(model, model_params)
    trace_count = len(traces)
    for idx in range(trace_count):
        model_results = compute_best_multivariate_results_for_trace(
            traces[idx], models, models_count)
        result_dict[traces[idx].get_trace_id()] = model_results
        log_modeling_progress(idx, trace_count, verbose)

def log_modeling_progress(trace_idx, trace_count, verbose=True):
    """Logs the modeling progress if `verbose` is set to True.

    The model progress prints a message to the screen indicating how many
    traces have been modeled by the current process based on `trace_idx`
    and `trace_count`.

    Parameters
    ----------
    trace_idx: int
        An integer representing the index of the trace that has finished
        processing.
    trace_count: int
        An integer representing the total number of traces to be processed.
     verbose: bool
        A boolean indicating whether the log will be printed.

    Returns
    -------
    None

    """
    if verbose:
        print("Process {0}: {1} of {2} traces modeled".format(
            os.getpid(), trace_idx + 1, trace_count))


def process_model_results_dict(model_results_dict):
    """Processes `model_results_dict` to a list.

    Parameters
    ----------
    model_results_dict: dict
        A dictionary containing model results. The keys are strings
        representing names of the target variables being modeled and the
        corresponding value is a list of `ModelResults` objects associated
        with that variable.

    Returns
    -------
    list
        A list containing the list representation of each `ModelResults`
        object in `model_results_dict`.

    """
    model_results_lst = []
    for model_var in model_results_dict.keys():
        for model_results in model_results_dict[model_var]:
            model_results_lst.extend(model_results.to_list())
    return model_results_lst

def get_results_list_from_trace_model_results(trace_model_results):
    """Converts `trace_model_results` to a results list

    A results list is a list of lists containing a list for each trace.
    These lists consist of the trace id followed by list representations of
    the `ModelResults` in `trace_model_results` associated with the trace.

    Parameters
    ----------
    trace_model_results: dict
        A dictionary in which each key is a string representing the id of a
        trace. The corresponding value is a list of `ModelResults`
        corresponding to the model results for that trace.

    Returns
    -------
    list
        A list of lists representing the results list.

    """
    results_lst = []
    for trace_id in trace_model_results.keys():
        trace_lst = [trace_id]
        for model_results in trace_model_results[trace_id]:
            trace_lst.extend(model_results.to_list())
        results_lst.append(trace_lst)
    return results_lst


def get_results_list_from_trace_model_dicts(trace_model_dicts):
    """Builds a results list from `trace_model_dicts`.

    The results list is a list of lists in which each list represents the
    modelling result for a trace. The list consists of the trace's ID followed
    by a list representation of all of the model results for that trace.

    Parameters
    ----------
    trace_model_dicts: dict
        A dictionary containing model results for each trace. The keys are
        strings representing trace IDs and the corresponding value is a
        dictionary of `ModelResults` for the trace. This nested dictionary has
        a key for each variable being modeled and the corresponding value is
        a list of `ModelResults` objects for that variable.

    Returns
    -------
    list
        A list of lists representing the results list.

    """
    results_lst = []
    for trace_id in trace_model_dicts.keys():
        results_lst.append([trace_id] + process_model_results_dict(
            trace_model_dicts[trace_id]))
    return results_lst


def process_and_output_model_results(model_results, models_count,
                                     model_name, output_dir):
    """Processes `model_results` and outputs them to `output_dir`.

    Parameters
    ----------
    model_results: dict
        A dictionary in which each key is a string representing the id of a
        trace. The corresponding value is a list of `ModelResults`
        corresponding to the model results for that trace.
    models_count: int
        An integer representing the number of best models kept for each trace.
    model_name: str
        A string specifying the name of the model.
    output_dir: str
        A string specifying the directory to which the model results will be
        output.

    Returns
    -------
    None

    Side Effect
    -----------
    The results in `model_results` are saved to a csv file containing one row
    per trace. The file is output to a file named `<name>_results.csv` in
    `output_dir`, where `<name>` is `model_name`.

    """
    results = get_results_list_from_trace_model_results(model_results)
    cols = get_col_list_for_params(
        range(1, models_count + 1), model_name,
        ModelResults.get_model_results_cols())
    output_model_results(
        results, ["id"] + cols, output_dir, "{}_results".format(model_name))


def process_and_output_multivariate_results(
    model_results, models_count, model_name, model_vars, output_dir):
        """Processes `model_results` and outputs them to `output_dir`.

        Parameters
        ----------
        model_results: dict
            A dictionary containing model results for each trace. The keys are
            strings representing trace IDs and the corresponding value is a
            dictionary of `ModelResults` for the trace. This nested dictionary
            has a key for each variable being modeled and the corresponding
            value is a list of `ModelResults` objects for that variable.
        models_count: int
            An integer representing the number of best models kept for each trace.
        model_name: str
            A string specifying the name of the model.
        model_vars: list
            A list of strings representing the names of the target variables
            in the multivariate modeling process.
        output_dir: str
            A string specifying the directory to which the model results will be
            output.

        Returns
        -------
        None

        Side Effect
        -----------
        The results in `model_results` are saved to a csv file containing one row
        per trace. The file is output to a file named `<name>_results.csv` in
        `output_dir`, where `<name>` is `model_name`.

        """
        results = get_results_list_from_trace_model_dicts(model_results)
        cols = get_col_list_for_params(
            range(1, models_count + 1), model_name,
            ModelResults.get_model_results_cols())
        cols = ["id"] + ["{0}_{1}".format(col, model_var)
                         for model_var, col in product(model_vars, cols)]
        output_model_results(
            results, cols, output_dir, "{}_results".format(model_name))



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
    model_params = get_univariate_model_build_input_params()
    model_args = (model_params['train_prop'], model_params['max_mem'])
    model_results = parallel.perform_trace_modelling(
        model_params['traces'], modeling_func, model_args)
    process_and_output_model_results(
        model_results, models_count, model_name, model_params['output_dir'])


def run_best_multivariate_models_for_all_traces(modeling_func, models_count,
                                                model_name, model_vars):
    """Models all traces using `modeling_func` with `model_name`.

    A series of multivariate models are run on each trace, and the best
    `models_count` of these models for each trace are recorded. `modeling_func`
    is used to run the models on a batch of traces and this function is
    parallelized to cover all traces. The best models are then retrieved.

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
    model_vars: list
        A list of the variables being modelled.

    Returns
    -------
    None

    Side Effect
    -----------
    The results of the modelling procedure is saved to a csv file containing
    one row per trace. The file is output to a file named `<name>_results.csv`
    where `<name>` is `model_name`.

    """
    model_params = get_multivariate_model_build_input_params()
    model_args = (model_params['train_prop'],)
    model_results = parallel.perform_trace_modelling(
        model_params['traces'], modeling_func, model_args)
    process_and_output_multivariate_results(
        model_results, models_count, model_name,
        model_vars, model_params['output_dir'])


def build_model_results_from_results_dict(model_results_dict):
    """Builds a `ModelResults` object from `model_results_dict`.

    Parameters
    ----------
    model_results_dict: pd.DataFrame
        A pandas DataFrame containing the model results for a trace.

    Returns
    -------
    ModelResults
        The `ModelResults` object obtained from `model_results_dict`.

    """
    results_dict = {k: v for k, v in model_results_dict.items()
                    if k not in ["id", "params"]}
    return ModelResults(model_results_dict['params'], results_dict)


def model_results_df_to_dict(model_results_df, model_name):
    """Converts `model_results_df` to a dictionary representation.

    The dictionary representation has a key for every trace in
    `model_results_df`. The corresponding value is a dictionary with two
    key-value pairs: one identifying the model as `model_name` and the other
    containing a `ModelResults` object summarizing the model results for that
    trace based on the data in `model_results_df`.

    Parameters
    ----------
    model_results_df: pd.DataFrame
        A pandas DataFrame containing the model results for all traces for the
        model specified by `model_name`.
    model_name: str
        A string representing the name of the model.

    Returns
    -------
    dict
        A dictionary representation of `model_results_df`.

    """
    model_results_dict = {}
    results_df = utils.process_model_results_df(model_results_df)
    for row_dict in results_df.to_dict(orient='records'):
        model_results = build_model_results_from_results_dict(row_dict)
        model_results_dict[row_dict['id']] = {'model': model_name,
                                              'results': model_results}
    return model_results_dict


def get_best_model_dict_for_trace(best_results_dict, row_dict, model_name):
    """Gets the best model dict from `best_results_dict` and `row_dict`.

    For the trace in `row_dict`, if the trace is not in `best_results_dict`
    then the results from `row_dict` are used as the best results for the
    trace. Otherwise, the best `ModelResults` for the trace is the better
    `ModelResults` object between `best_results_dict` and `row_dict`. The
    dictionary has the model name of the best model and the corresponding
    `ModelResults` object.

    Parameters
    ----------
    best_results_dict: dict
        A dictionary containing the best model results for each trace. The
        keys are strings representing trace IDs. The corresponding value is
        a dictionary with 2 key-value pairs: one specifying the name of the
        best model and the other specifying the `ModelResults` object
        associated with the model.
    row_dict: dict
        A dictionary specifying the model results for a model of type
        `model_name` built on a trace.
    model_name: str
        A string specifying the type of model to which the results in row_dict
        refer.

    Returns
    -------
    dict
        The dictionary of best results for the trace based on `row_dict` and
        `best_results_dict`.

    """
    model_results = build_model_results_from_results_dict(row_dict)
    if row_dict['id'] not in best_results_dict:
        return {'model': model_name, 'results': model_results}
    old_results = best_results_dict[row_dict['id']]['results']
    if model_results.is_better(old_results):
        return {'model': model_name, 'results': model_results}
    return best_results_dict[row_dict['id']]


def update_best_model_results_dict(best_results_dict,
                                   new_results_df, model_name):
    """Updates `best_results_dict` with the results from `new_results_df`.

    For each record in `new_results_df`, the corresponding trace is updated
    in `best_results_dict` if the model results in `new_results_df` are better
    than the model results in `best_results_dict` for that trace.

    Parameters
    ----------
    best_results_dict: dict
        A dictionary containing the best model results for each trace. The
        keys are strings representing trace IDs. The corresponding value is
        a dictionary with 2 key-value pairs: one specifying the name of the
        best model and the other specifying the `ModelResults` object
        associated with the model.
    new_results_df: pd.DataFrame
        A pandas DataFrame with new model results which is being used to
        update `best_results_dict`.
    model_name: str
        A string representing the name of the model which generated the results
        in `new_results_df`.

    Returns
    -------
    dict
        The dictionary obtained from `best_results_dict` after updating based
        on the data in `new_results_df`.

    """
    results_df = utils.process_model_results_df(new_results_df)
    for row_dict in results_df.to_dict(orient='records'):
        best_results_dict[row_dict['id']] = get_best_model_dict_for_trace(
            best_results_dict, row_dict, model_name)
    return best_results_dict


def get_best_model_results_dict_from_results_dfs(model_results_dfs):
    """Gets a dictionary of the best model results from `model_results_dfs`.

    A dictionary containing the best `ModelResults` object for each trace
    is built from the model results dataframes in `model_results_dfs`. The
    keys are strings representing trace IDs and the corresponding value is
    a dictionary of 2 key-value pairs: specifying the name of the best model
    and the associated `ModelResults` for that trace.

    Parameters
    ----------
    model_results_dfs: dict
        A dictionary of model results DataFrames where the keys are strings
        representing the names of the model to which the DataFrame refers to.
        Each dataframe contains the results of the corresponding model fit on
        each trace.

    Returns
    -------
    dict
        The dictionary of the best `ModelResults` for each trace obtained from
        `model_results_dfs`.

    """
    model_names = list(model_results_dfs.keys())
    best_results_dict = model_results_df_to_dict(
        model_results_dfs[model_names[0]], model_names[0])
    for model_name in model_names[1:]:
        best_results_dict = update_best_model_results_dict(
            best_results_dict, model_results_dfs[model_name], model_name)
    return best_results_dict


def output_best_model_results_dict(best_results_dict, output_dir):
    """Outputs `best_results_dict` to a `csv` file.

    Parameters
    ----------
    best_results_dict: dict
        A dictionary containing the best model results for each trace. The
        keys are strings representing trace IDs. The corresponding value is
        a dictionary with 2 key-value pairs: one specifying the name of the
        best model and the other specifying the `ModelResults` object
        associated with the model.
    output_dir: str
        A string representing the name of the output directory to which the
        results are output.

    Returns
    -------
    None

    Side Effect
    -----------
    A file named `best_model_results.csv` is saved to `output_dir` containing
    the results from `best_results_dict`.

    """
    best_results_lst = [
        [trace_id, model_dict['model']] + model_dict['results'].to_list()
         for trace_id, model_dict in best_results_dict.items()]
    cols = ["{}_best".format(col_name) for col_name
            in ModelResults.get_model_results_cols()]
    output_model_results(best_results_lst, ["id", "model"] + cols,
                         output_dir, "best_model_results")


def output_best_model_results_from_model_results_dfs(model_results_dfs,
                                                     output_dir):
    """Outputs the best model results based on `model_results_dfs`.

    For each trace, the best `ModelResults` is used across all the model
    results DataFrame in `model_results_dfs`.

    Parameters
    ----------
    model_results_dfs: dict
        A dictionary of model results DataFrames where the keys are strings
        representing the names of the model to which the DataFrame refers to.
        Each dataframe contains the results of the corresponding model fit on
        each trace.
    output_dir: str
        A string representing the name of the output directory to which the
        results are output.

    Returns
    -------
    None

    Side Effect
    -----------
    A file named `best_model_results.csv` is saved to `output_dir` containing
    the best model results for each trace from `model_results_dfs`.

    """
    model_results_dict = get_best_model_results_dict_from_results_dfs(
        model_results_dfs)
    output_best_model_results_dict(model_results_dict, output_dir)


def get_percentiles_df_for_model_results(model_results_dict,
                                         model_names_lst, analysis_col):
    """The percentile dataframe for `analysis_col` in `model_results_dict`.

    The percentile dataframe is a pandas DataFrame containing the 1, 25, 50,
    75, and 99 percentile values for `analysis_col` for each model in
    `model_names_lst` based on the data in `model_results_dict`.

    Parameters
    ----------
    model_results_dict: dict
        A dictionary of model results DataFrames where the keys are strings
        representing the names of the model to which the DataFrame refers to.
        Each dataframe contains the results of the corresponding model fit on
        each trace.
    model_names_lst: list
        A list of strings representing the names of models for which the
        corresponding model results dataframe appears in `model_results_dict`.
    analysis_col: str
        A string representing the name of the column in the model results
        dataframes for which the percentiles are calculated.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame representing the percentile dataframe for
        `analysis_col` based on the data in `model_results_dict`.

    """
    results_lst = []
    for model_name in model_names_lst:
        col_name = "{0}_{1}".format(analysis_col, model_name)
        results_lst.append(list(np.percentile(
            model_results_dict[model_name][col_name].values,
            [1, 25, 50, 75, 99])))
    return pd.DataFrame(results_lst, index=model_names_lst,
                        columns=["P1", "P25", "Median", "P75", "P99"])
