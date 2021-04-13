"""A set of helper functions used in the analysis process.

"""
import sys
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import product
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


def get_model_stats_for_trace(data_trace, models):
    """Gets statistics from `models` for `data_trace`.

    For each model in `models`, the model is fit to `data_trace` and
    the mean absolute scaled error on the test set is computed.

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
            "train_prop": float(sys.argv[5]),
            "aggregation_window": int(sys.argv[6])}


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
        params['input_dir'], params['file_id'],
        params['min_trace_length'], params["aggregation_window"])
    traces = trace_handler.run_processing_pipeline()
    return traces, params['output_dir'], params['train_prop']


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


def model_traces_and_evaluate(model, model_params, traces,
                              results_lst, verbose=True):
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
    verbose: bool
        A boolean indicating whether the log will be printed.

    Returns
    -------
    None

    """
    models = build_models_from_params_list(model, model_params)
    trace_count = len(traces)
    for idx in range(trace_count):
        results_lst.append(get_model_stats_for_trace(traces[idx], models))
        log_modeling_progress(idx, trace_count, verbose)

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


def is_better_model(new_mase, new_under_mase, old_mase, old_under_mase):
    """Determines if the new model performs better than the old model.

    The new model performs better than the old model if the weighted
    difference of the MASEs is positive.

    """
    under_mase_diff = old_under_mase - new_under_mase
    mase_diff = old_mase - new_mase
    w = specs.OVERALL_MASE_WEIGHT
    return ((under_mase_diff >= 0 and mase_diff >= 0) or
            (under_mase_diff >= 0 and under_mase_diff >= -w * mase_diff) or
            (mase_diff >= 0 and w * mase_diff > -under_mase_diff))


def update_with_model_results(results_lst, model_results, cutoff):
    """Updates `results_lst` with the model results.

    If `results_lst` has fewer than `cutoff` entries or the test MASE is lower
    than the test MASE of a model already contained in `results_lst`, then
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
        old_model_idx = 2 + (len(model_results) * idx)
        is_better = is_better_model(
            model_results[2], model_results[3],
            results_lst[old_model_idx + 1], results_lst[old_model_idx + 2])
        if is_better:
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
    test MASE when built on `trace`.

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
    cutoff = (len(specs.MODELING_COLS) * models_count) + 1
    for model in models:
        best_results = update_model_stats_for_trace(
            trace, model, best_results, cutoff)
    return pad_list(best_results, np.nan, cutoff)

def get_best_multivariate_models_for_trace(trace, models, models_count):
    """Gets stats for the best multivariate models in `models` for `trace`.

    The best models in `models` are the `models_count` models with the lowest
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
    dict
        A dictionary consisting of a list for each variable being modeled.
        Each list consists of the ID of `trace` followed by the results for the
        `models_count` best models from `models` fit to `trace`.

    """
    cutoff = (len(specs.MODELING_COLS) * models_count) + 1
    for model in models:
        best_results = update_model_stats_for_trace(
            trace, model, best_results, cutoff)
    for model_var in best_results.keys():
        best_results[model_var] = pad_list(
            best_results[model_var], np.nan, cutoff)
    return best_results

def get_best_model_results_for_traces(model, model_params, traces,
                                      result_lst, models_count, verbose=True):
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
        result_lst.append(
            get_best_models_for_trace(traces[idx], models, models_count))
        log_modeling_progress(idx, trace_count, verbose)

def add_model_results_to_dict(trace, models, models_count, result_dict):
    """Adds the model results of `models` fit on `trace` to `result_dict`.

    Parameters
    ----------
    trace: Trace
        The `Trace` being modeled.
    models: list
        A list of models to be fit to `trace`.
    models_count: int
        An integer representing the number of model results to save to
        `result_dict`
    result_dict: dict
        A dictionary storing model results for each trace.

    Returns
    -------
    None

    """
    model_results = get_best_multivariate_models_for_trace(
        trace, models, models_count)
    for model_var in model_results.keys():
        results_dict[model_var].append(model_results[model_var])

def get_best_multivariate_model_results_for_traces(
    model, model_params, traces, result_dict, models_count, verbose=True):
    """Gets the `models_count` best model results for the traces of `traces`.

    For each trace in `traces` a `model` object is built for each set of model
    parameters in `model_params` and the results of the best `models_count`
    models are saved in `result_dict`. If fewer than `models_count` models are
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
        add_model_results_to_dict(
            traces[idx], models, models_count, result_dict)
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
    results = parallel.perform_trace_modelling(
        traces, modeling_func, train_prop)
    cols = get_col_list_for_params(
        model_params, model_name,
        ["train_mase", "test_mase", "prop_under_preds",
        "max_under_pred", "prop_over_preds", "avg_over_pred"])
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
    results = parallel.perform_trace_modelling(
        traces, modeling_func, train_prop)
    cols = get_col_list_for_params(
        range(1, models_count + 1), model_name, specs.MODELING_COLS)
    output_model_results(
        results, ["id"] + cols, output_dir, "{}_results".format(model_name))

def plot_cumulative_distribution_function(dist_vals, ax, title, color, desc):
    """Plots the cumulative distribution of `dist_vals`.

    Parameters
    ----------
    dist_vals: np.array
        A numpy array representing the values of the distribution for
        which the cumulative distribution function is generated.
    ax: plt.axis
        The axis on which the cumulative distribution is rendered.
    title: str
        A string representing the title for the distribution function.
    color: str
        A string representing the color for the plot.
    desc: str
        A string describing the type of CDF.

    Returns
    -------
    None

    """
    dist_vals[np.isnan(dist_vals)] = 0
    x_vals = np.sort(dist_vals)
    pdf = x_vals / np.sum(x_vals)
    cdf = np.cumsum(pdf)
    ax.plot(x_vals, cdf, color=color)
    ax.set_title("{0} of Maximum Memory vs {1}".format(desc, title))
