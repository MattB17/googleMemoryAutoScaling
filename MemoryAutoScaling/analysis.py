"""A set of helper functions used in the analysis process.

"""
import sys
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import product
from MemoryAutoScaling.Evaluation import ModelResults, HarvestStats
from MemoryAutoScaling import parallel, plotting, specs, utils
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


def build_models_from_params_list(trace_model, params_lst):
    """Builds `trace_model` objects from the params in `params_lst`.

    A separate `trace_model` object is built for each set of params
    in `params_lst`.

    Parameters
    ----------
    trace_model: TraceModel.class
        A reference to a `TraceModel` class representing the models being
        created.
    params_lst: list
        A list of dictionaries in which each dictionary represents the set of
        named parameters used to initialize a `trace_model`.

    Returns
    -------
    list
        A list of `trace_model` objects instantiated from the parameters
        in `params_lst`.

    """
    return [trace_model(**params) for params in params_lst]


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
            "val_prop": float(sys.argv[6]),
            "aggregation_window": int(sys.argv[7]),
            "max_mem": (sys.argv[8].lower() == "true")}


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
            'val_prop': params['val_prop'],
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
    new_model_results = model.run_model_pipeline_for_trace(trace, tuning=True)
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
    new_model_results_dict = model.run_model_pipeline_for_trace(
        trace, tuning=True)
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


def compute_best_results_for_trace(model_type, trace, models,
                                   models_count, fixed_params):
    """Gets `ModelResults` for the best models of `models` for `trace`.

    The best models of `models` are the `models_count` models with the lowest
    test MASE when built on `trace`.

    Parameters
    ----------
    model_type: TraceModel.class
        A reference to a `TraceModel` class representing the types of the
        models being fit to `trace`.
    trace: Trace
        The `Trace` object being modeled.
    models: list
        A list of `TraceModel` objects representing the models being fit
        to `trace` and from which the best models are chosen.
    models_count: int
        An integer representing the number of models to include in the results.
    fixed_params: dict
        A dictionary of model parameters representing the parameters that are
        common to each model being fit to `trace`, given by `models`.

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
    test_results = get_test_results_from_val_results_list(
        model_type, best_results, fixed_params, trace)
    return pad_model_results(test_results, models_count)

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

def get_best_model_results_for_traces(trace_model, model_params, traces,
                                      result_dict, models_count,
                                      fixed_model_params, verbose=True):
    """Gets the `models_count` best model results for the traces of `traces`.

    For each trace in `traces` a `trace_model` object is built for each set of
    model parameters in `model_params` and the results of the best
    `models_count` models are saved in `result_dict`.

    Parameters
    ----------
    trace_model: TraceModel.class
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
    fixed_model_params: dict
        A dictionary of model parameters representing the parameters that are
        common to each model being fit to `trace`.
    verbose: bool
        A boolean indicating whether to run in verbose mode. The default
        value is True, in which case progress is printed to the screen.

    Returns
    -------
    None

    """
    models = build_models_from_params_list(trace_model, model_params)
    trace_count = len(traces)
    for idx in range(trace_count):
        model_results = compute_best_results_for_trace(
            trace_model, traces[idx], models, models_count, fixed_model_params)
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
    cols = get_col_list_for_params(range(1, models_count + 1), model_name,
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
        cols = get_col_list_for_params(range(1, models_count + 1), model_name,
                                       ModelResults.get_model_results_cols())
        output_model_results(results, ["id"] + cols, output_dir,
                             "{}_results".format(model_name))



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
    params = get_univariate_model_build_input_params()
    model_args = (params['train_prop'], params['val_prop'], params['max_mem'])
    model_results = parallel.perform_trace_modelling(
        params['traces'], modeling_func, model_args)
    process_and_output_model_results(
        model_results, models_count, model_name, params['output_dir'])


def get_univariate_model_results(traces, modeling_func, model_args,
                                 models_count, model_name, output_dir):
    """Models `traces` with the univariate model from `modeling_func`.

    A series of models are run on each trace, and the best `models_count` of
    these models for each trace are recorded. `modeling_func` is used to run
    the models on a batch of traces and this function is parallelized to cover
    all traces. The best models are then retrieved.

    Parameters
    ----------
    traces: list
        A list of `Trace` objects to be modeled.
    modeling_func: function
        The function used to perform the modelling on a batch of traces. The
        function takes three arguments: a list of `Trace` objects on which the
        models are run, a list to which results are saved, and a float
        specifying the proportion of data in the training set.
    model_args: tuple
        A tuple containing the arguments used for `model_func`.
    models_count: int
        An integer representing the number of models to record. The results
        for the `models_count` best models for each trace are recorded.
    model_name: str
        A string specifying the name of the model.
    output_dir: str
        A string specifying the directory to which the results will be output.

    Returns
    -------
    None

    Side Effect
    -----------
    The results of the modelling procedure is saved to a csv file containing
    one row per trace. The file is output to a file named `<name>_results.csv`
    where `<name>` is `model_name`.

    """
    model_results = parallel.perform_trace_modelling(
        traces, modeling_func, model_args)
    process_and_output_model_results(
        model_results, models_count, model_name, output_dir)

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


def build_harvest_stats_dict_from_model_results(model_results_dict):
    """A dictionary of harvest statistics from `model_results_dict`.

    Parameters
    ----------
    model_results_dict: pd.DataFrame
        A pandas DataFrame containing the model results for a trace.

    Returns
    -------
    dict
        A dictionary containing `HarvestStats` objects for the harvest
        statistics in `model_results_dict`.

    """
    harvest_dict = {}
    harvest_cols = HarvestStats.get_harvest_stat_columns()
    for buffer_pct in specs.BUFFER_PCTS:
        harvest_dict[buffer_pct] = HarvestStats(
            model_results_dict["{0}_{1}".format(harvest_cols[0], buffer_pct)],
            model_results_dict["{0}_{1}".format(harvest_cols[1], buffer_pct)])
    return harvest_dict


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
    harvest_dict = build_harvest_stats_dict_from_model_results(
        model_results_dict)
    results_dict = {k: v for k, v in model_results_dict.items()
                    if k not in ["id", "params"]}
    return ModelResults(
        model_results_dict['params'], results_dict, harvest_dict)


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
        x, _ = plotting.get_cdf_values(
            model_results_dict[model_name][col_name].values)
        results_lst.append(list(np.percentile(x, [1, 25, 50, 75, 99])))
    return pd.DataFrame(results_lst, index=model_names_lst,
                        columns=["P1", "P25", "Median", "P75", "P99"])


def standardize_counts_dict(counts_dict):
    """Converts the numbers in `counts_dict` to percentages.

    Parameters
    ----------
    counts_dict: dict
        A dictionary of counts being standardized.

    Returns
    -------
    dict
        The dictionary obtained from `counts_dict` after converting each
        count to a percentage.

    """
    sum = 0
    for count_key in counts_dict.keys():
        sum += counts_dict[count_key]
    for count_key in counts_dict.keys():
        counts_dict[count_key] = round(counts_dict[count_key] / sum * 100, 2)
    return counts_dict


def get_stationary_1_diff_count(stats_df):
    """The number of stationary traces in `stats_df` after 1 differences.

    Parameters
    ----------
    stats_df: pd.DataFrame
        The pandas DataFrame from which the stationary results are calculated.

    Returns
    -------
    int
        An integer representing the number of traces that were stationary
        after one level of differencing based on data in `stats_df`.

    """
    return len(stats_df[((stats_df['adf_p_val_diff'] < 0.05) &
                        ~np.isnan(stats_df['adf_p_val_diff'])) &
                        ((stats_df['adf_p_val'] >= 0.05) |
                        np.isnan(stats_df['adf_p_val']))])


def get_stationary_2_diff_count(stats_df):
    """The number of stationary traces in `stats_df` after 2 differences.

    Parameters
    ----------
    stats_df: pd.DataFrame
        The pandas DataFrame from which the stationary results are calculated.

    Returns
    -------
    int
        An integer representing the number of traces that were stationary
        after two levels of differencing based on data in `stats_df`.

    """
    return len(stats_df[((stats_df['adf_p_val_diff2'] < 0.05) &
                        ~np.isnan(stats_df['adf_p_val_diff2'])) &
                        ((stats_df['adf_p_val_diff'] >= 0.05) |
                        np.isnan(stats_df['adf_p_val_diff'])) &
                        ((stats_df['adf_p_val'] >= 0.05) |
                        np.isnan(stats_df['adf_p_val']))])


def get_non_stationary_count(stats_df):
    """The number of non-stationary traces in `stats_df`.

    A trace is deemed non-stationary if it is still not stationary after
    2 levels of differencing.

    Parameters
    ----------
    stats_df: pd.DataFrame
        The pandas DataFrame from which the stationary results are calculated.

    Returns
    -------
    int
        An integer representing the number of traces that were not stationary
        after two levels of differencing based on data in `stats_df`.

    """
    return len(stats_df[((stats_df['adf_p_val_diff2'] >= 0.05) |
                        np.isnan(stats_df['adf_p_val_diff2'])) &
                        ((stats_df['adf_p_val_diff'] >= 0.05) |
                        np.isnan(stats_df['adf_p_val_diff'])) &
                        ((stats_df['adf_p_val'] >= 0.05) |
                        np.isnan(stats_df['adf_p_val']))])

def stationary_results_from_stats_df(stats_df):
    """Retrieves results of stationarity tests from `stats_df`.

    It is assumed that `stats_df` has columns "adf_p_val", "adf_p_val_diff",
    and "adf_p_val_diff2". The stationary results report the percent of traces
    that are stationary after 0, 1, or 2 levels of differencing, and the traces
    that are not stationary after 2 levels of differencing.

    Parameters
    ----------
    stats_df: pd.DataFrame
        The pandas DataFrame from which the stationary results are calculated.

    Returns
    -------
    dict
        A dictionary containing the stationary results.

    """
    results_dict = {}
    results_dict['diff_0'] = len(stats_df[stats_df['adf_p_val'] < 0.05])
    results_dict['diff_1'] = get_stationary_1_diff_count(stats_df)
    results_dict['diff_2'] = get_stationary_2_diff_count(stats_df)
    results_dict['other'] = get_non_stationary_count(stats_df)
    return standardize_counts_dict(results_dict)


def get_stationary_results_df(stats_dfs):
    """A dataframe of stationary results from `stats_dfs`.

    Parameters
    ----------
    stats_dfs: dict
        A dictionary of pandas DataFrames from which the stationary results
        are calculated.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing a summary of the stationary results from
        `stats_dfs`.

    """
    results_dict = {'diff_0': [], 'diff_1': [], 'diff_2': [], 'other': []}
    indices = []
    for stats_name in stats_dfs.keys():
        indices.append(" ".join(stats_name.split("_")).title())
        stats_dict = stationary_results_from_stats_df(stats_dfs[stats_name])
        results_dict['diff_0'].append("{}%".format(stats_dict['diff_0']))
        results_dict['diff_1'].append("{}%".format(stats_dict['diff_1']))
        results_dict['diff_2'].append("{}%".format(stats_dict['diff_2']))
        results_dict['other'].append("{}%".format(stats_dict['other']))
    df = pd.DataFrame(results_dict)
    df.columns = ["Stationary", "Stationary Diff 1",
                  "Stationary Diff 2", "Non-Stationary"]
    df.index = indices
    return df


def get_total_utilization_across_traces(stat_df):
    """Gets total utilization across the traces in `stat_df`.

    Parameters
    ----------
    stat_df: pd.DataFrame
        A pandas DataFrame containing the utilization statistics for the
        traces.

    Returns
    -------
    str
        A string representing the percent utilization across all traces in
        `stat_df`, rounded to 2 decimal places.

    """
    utilization_vals = utils.impute_for_time_series(
        stat_df['utilization'].values, 0.0)
    allocated_vals = utils.impute_for_time_series(
        stat_df['allocated'].values, 0.0)
    numerator = np.sum(utilization_vals * allocated_vals)
    denominator = np.sum(allocated_vals)
    return "{}%".format(round(100 * (numerator / denominator), 2))


def get_total_utilization_df(stats_dfs):
    """A dataframe of total utilization rates from `stats_dfs`.

    Parameters
    ----------
    stats_dfs: dict
        A dictionary of pandas DataFrames from which the stationary results
        are calculated.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the total utilization rate for each
        DataFrame in `stats_dfs`.

    """
    results_lst = []
    indices = []
    for stats_name in stats_dfs.keys():
        indices.append(" ".join(stats_name.split("_")).title())
        results_lst.append(
            get_total_utilization_across_traces(stats_dfs[stats_name]))
    df = pd.DataFrame({"Total Utilization": results_lst})
    df.index = indices
    return df


def build_model_from_model_results(model, model_results, other_params):
    """Builds a `model` object from `model_results` and `other_params`.

    Parameters
    ----------
    model: TraceModel.class
        A reference to a `TraceModel` class representing the model being
        created.
    model_results: ModelResults
        A `ModelResults` object containing the model results used to build
        `model`.
    other_params: dict
        A dictionary of other parameters used to build `model`.

    Returns
    -------
    TraceModel
        A `TraceModel` object of type `model` built from `model_results` and
        `other_params`.

    """
    model_params = {**model_results.get_model_params(), **other_params}
    return model(**model_params)


def get_test_results_from_val_results(trace_model, val_results,
                                      other_params, trace):
    """The test results for `trace` based on the validation model.

    The validation model is a model of type `trace_model`, which is
    constructed based on `val_results` and `other_params`. Once constructed,
    the model is run on trace for the test set and the model results are
    calculated.

    Parameters
    ----------
    trace_model: TraceModel.class
        A reference to a `TraceModel` class representing the model built
        from the validation results and evaluated on the test set.
    val_results: ModelResults
        A `ModelResults` object representing the results of running a model of
        type `trace_model` on the validation set of `trace`.
    other_params: dict
        A dictionary of other parameters used to build `model`.
    trace: Trace
        The `Trace` object that is being modeled.

    Returns
    -------
    ModelResults
        A `ModelResults` object representing the results of training a model
        of type `trace_model` on `trace` where the model is constructed based
        on the validation results.

    """
    model_for_trace = build_model_from_model_results(
        trace_model, val_results, other_params)
    return model_for_trace.run_model_pipeline_for_trace(trace, tuning=False)


def get_test_results_from_val_results_list(trace_model, val_results_lst,
                                           other_params, trace):
    """The test results for `trace` based on the validation models.

    A validation model is a model of type `trace_model`. There is one
    validation model for each `ModelResults` object in `val_results_lst`,
    which is constructed based on the `ModelResults` and `other_params`.
    Each such model is evaluated on the test set of `trace` to get a new
    `ModelResults` object for the test set.

    Parameters
    ----------
    trace_model: TraceModel.class
        A reference to a `TraceModel` class representing the types of models
        that will be built from the validation results and evaluated on the
        test set.
    val_results_lst: list
        A list of `ModelResults` objects representing the results of running
        models of type `trace_model` on the validation set of `trace`.
    other_params: dict
        A dictionary of other parameters used to build the models. These
        parameters are the same across every model constructed and do not
        depend on the validation results.
    trace: Trace
        The `Trace` object that is being modeled.

    Returns
    -------
    list
        A list of `ModelResults` objects representing the results of
        evaluating the validation models of type `trace_model` on the test
        set of `trace`.

    """
    test_results_lst = []
    for val_results in val_results_lst:
        test_results_lst.append(get_test_results_from_val_results(
            trace_model, val_results, other_params, trace))
    return test_results_lst

def calculate_prop_harvested_for_model(model_results_df,
                                       model_name, buffer_pct):
    """The proportion harvested across all traces with `model_name`.

    Parameters
    ----------
    model_results_df: pd.DataFrame
        A pandas DataFrame containing the model results for each trace.
    model_name: str
        A string representing the name of the model to which the results
        apply.
    buffer_pct: float
        A float representing the buffer percentage applied to predictions.

    Returns
    -------
    float
        A float in the range [0, 1] representing the proportion harvested.

    """
    spare_col = 'total_spare_{}'.format(model_name)
    harvest_col = 'prop_harvested_{0}_{1}'.format(buffer_pct, model_name)
    spare_vals = model_results_df[spare_col].fillna(0).values
    harvested_props = model_results_df[harvest_col].fillna(0).values
    return np.sum(spare_vals * harvested_props) / np.sum(spare_vals)


def get_prop_harvested_dict_for_models(model_results_dfs, buffer_pct):
    """The proportion harvested for each model in `model_results_dfs`.

    For each model results DataFrame in `model_results_dfs`, the proportion
    of total resource harvested across all traces is calculated. The
    dictionary has a key as a string representing the model name and the
    corresponding value is a float representing the proportion harvested by
    that model across all traces.

    Parameters
    ----------
    model_results_dfs: dict
        A dictionary of model results DataFrames where the keys are strings
        representing the model name and the corresponding value is a
        DataFrame containing the associated model results.
    buffer_pct: float
        A float representing the buffer percentage applied to predictions.

    Returns
    -------
    dict
        A dictionary of proportion of resources harvested for each model.

    """
    prop_harvested_dict = {}
    for model_name in model_results_dfs.keys():
        prop_harvested_dict[model_name] = calculate_prop_harvested_for_model(
            model_results_dfs[model_name], model_name, buffer_pct)
    return prop_harvested_dict


def plot_prop_harvested_by_model(model_results_dfs, buffer_pct):
    """Plots the proportion harvested for each model.

    `model_results_dfs` contains a DataFrame for each model recording the
    model results for each trace for that model, after tuning the parameters
    for each trace. The proportion of the resource harvested across all
    traces is calculated for each model and plotted.

    Parameters
    ----------
    model_results_dfs: dict
        A dictionary of model results DataFrames
    buffer_pct: float
        A float representing the buffer percentage applied to predictions.

    Returns
    -------
    None

    """
    prop_harvested_dict = get_prop_harvested_dict_for_models(
        model_results_dfs, buffer_pct)
    title = "Harvested With {}% Prediction Buffer".format(
        int(buffer_pct * 100))
    plotting.plot_proportions_across_models(prop_harvested_dict, title)


def plot_usage_vs_allocated(trace, is_mem=True):
    """Plots the used vs allocated values of the resource for `trace`.

    Parameters
    ----------
    trace: Trace
        The `Trace` for which the usage data is plotted.
    is_mem: bool
        A boolean value indicating whether memory is the target resource.
        If true, memory is the target resource, otherwise it is CPU.

    Returns
    -------
    None

    """
    resource_col = specs.MAX_MEM_COL if is_mem else specs.MAX_CPU_COL
    allocated_amt = trace.get_amount_allocated_for_target(resource_col)
    usage_ts = trace.get_resource_usage(resource_col)
    resource_name = "Memory" if is_mem else "CPU"
    plotting.plot_usage_vs_allocated(usage_ts, allocated_amt, resource_name)
