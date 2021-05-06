"""Builds all univariate models for each trace and evaluates the performance
of each on the training and testing sets. The univariate models are moving
average, exponential smoothing, auto regression, ARIMA, linear regression,
support vector regression, and XGBoost regression.

"""
import pandas as pd
from itertools import product
from MemoryAutoScaling import analysis, specs, utils
from MemoryAutoScaling.Models.Sequential import TraceMovingAverage
from MemoryAutoScaling.Models.Sequential import TraceExponentialSmoothing
from MemoryAutoScaling.Models.Statistical import TraceAR
from MemoryAutoScaling.Models.Statistical import TraceARIMA
from MemoryAutoScaling.Models.ML import TraceRegression
from MemoryAutoScaling.Models.ML import TraceSVM
from MemoryAutoScaling.Models.ML import TraceXGB
from MemoryAutoScaling.DataHandling import MLDataHandler
pd.options.mode.chained_assignment = None


MA_WINDOWS = [1, 2, 3, 4, 5, 7, 10]


def build_ma_models_for_traces(trace_lst, results_dict, train_prop,
                               val_prop, max_mem):
    fixed_model_params = {'initial_pred': 0.0, 'train_prop': train_prop,
                          'val_prop': val_prop, 'max_mem': max_mem}
    ma_params_lst = [{'window_length': ma_win, **fixed_model_params}
                     for ma_win in MA_WINDOWS]
    analysis.get_best_model_results_for_traces(
        TraceMovingAverage, ma_params_lst, trace_lst,
        results_dict, specs.MODELS_COUNT, fixed_model_params)


ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]


def build_es_models_for_traces(trace_lst, results_dict,
                               train_prop, val_prop, max_mem):
    fixed_model_params = {'initial_pred': 0.001, 'train_prop': train_prop,
                          'val_prop': val_prop, 'max_mem': max_mem}
    es_params_lst = [{'alpha': alpha_val, **fixed_model_params}
                     for alpha_val in ALPHAS]
    analysis.get_best_model_results_for_traces(
        TraceExponentialSmoothing, es_params_lst, trace_lst,
        results_dict, specs.MODELS_COUNT, fixed_model_params)


AR_COMPS = [1, 2, 3, 4, 5, 7, 10]


def build_ar_models_for_traces(trace_lst, results_dict, train_prop,
                               val_prop, max_mem):
    fixed_model_params = {'train_prop': train_prop,
                          'val_prop': val_prop,
                          'max_mem': max_mem}
    ar_params_lst = [{'p': p, **fixed_model_params} for p in AR_COMPS]
    analysis.get_best_model_results_for_traces(
        TraceAR, ar_params_lst, trace_lst,
        results_dict, specs.MODELS_COUNT, fixed_model_params)


ARIMA_p = [p for p in range(5)]
ARIMA_d = [d for d in range(2)]
ARIMA_q = [q for q in range(5)]

MA_PARAMS = [(0, 0, q) for q in ARIMA_q[1:]]
AR_PARAMS = [(p, 0, 0) for p in ARIMA_p[1:]]
ARIMA_PARAMS = [arima_tup for arima_tup in product(ARIMA_p, ARIMA_d, ARIMA_q)
                if arima_tup not in MA_PARAMS + AR_PARAMS + [(0, 0, 0)]]


def build_arima_models_for_traces(traces_lst, results_dict,
                                  train_prop, val_prop, max_mem):
    fixed_model_params = {'train_prop': train_prop,
                          'val_prop': val_prop,
                          'max_mem': max_mem}
    arima_params_lst = [{'p': p, 'd': d, 'q': q, **fixed_model_params}
                        for p, d, q in ARIMA_PARAMS]
    analysis.get_best_model_results_for_traces(
        TraceARIMA, arima_params_lst, traces_lst,
        results_dict, specs.MODELS_COUNT, fixed_model_params)


FEATURE_COLS = specs.get_lagged_trace_columns(specs.LAGS)
REG_VALS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def build_reg_models_for_traces(trace_lst, results_dict,
                                train_prop, val_prop, max_mem):
    target_col = specs.get_target_variable(max_mem)
    data_handler = MLDataHandler(
        FEATURE_COLS, [target_col], train_prop, val_prop)
    fixed_model_params = {'data_handler': data_handler, 'lags': specs.LAGS}
    reg_params_lst = [{'reg_val': reg_val, **fixed_model_params}
                       for reg_val in REG_VALS]
    analysis.get_best_model_results_for_traces(
        TraceRegression, reg_params_lst, trace_lst,
        results_dict, specs.MODELS_COUNT, fixed_model_params)


SVM_VALS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def build_svm_models_for_traces(trace_lst, results_dict,
                                train_prop, val_prop, max_mem):
    target_col = specs.get_target_variable(max_mem)
    data_handler = MLDataHandler(
        FEATURE_COLS, [target_col], train_prop, val_prop)
    fixed_model_params = {'data_handler': data_handler, 'lags': specs.LAGS}
    svm_params_lst = [{'reg_val': reg_val, **fixed_model_params}
                       for reg_val in SVM_VALS]
    analysis.get_best_model_results_for_traces(
        TraceSVM, svm_params_lst, trace_lst,
        results_dict, specs.MODELS_COUNT, fixed_model_params)


LEARNING_RATES = [0.1, 0.3, 0.5]
ESTIMATORS = [10, 50, 100]
DEPTHS = [1, 2]


def build_xgb_models_for_traces(trace_lst, results_dict,
                                train_prop, val_prop, max_mem):
    target_col = specs.get_target_variable(max_mem)
    data_handler = MLDataHandler(
        FEATURE_COLS, [target_col], train_prop, val_prop)
    fixed_model_params = {'data_handler': data_handler, 'lags': specs.LAGS}
    xgb_params_lst = [{'learning_rate': learning_rate,
                       'estimators': n_estimators,
                       'depth': depth,
                       **fixed_model_params}
                      for learning_rate, n_estimators, depth
                      in product(LEARNING_RATES, ESTIMATORS, DEPTHS)]
    analysis.get_best_model_results_for_traces(
        TraceXGB, xgb_params_lst, trace_lst,
        results_dict, specs.MODELS_COUNT, fixed_model_params)


if __name__ == "__main__":
    params = analysis.get_univariate_model_build_input_params()
    model_args = (params['train_prop'], params['val_prop'], params['max_mem'])

    print("Build Moving Average Models")
    analysis.get_univariate_model_results(
        params['traces'], build_ma_models_for_traces, model_args,
        specs.MODELS_COUNT, "ma", params['output_dir'])

    print("Build Exponential Smoothing Models")
    analysis.get_univariate_model_results(
        params['traces'], build_es_models_for_traces, model_args,
        specs.MODELS_COUNT, "es", params['output_dir'])

    print("Build Auto Regressive Models")
    analysis.get_univariate_model_results(
        params['traces'], build_ar_models_for_traces, model_args,
        specs.MODELS_COUNT, "ar", params['output_dir'])

    print("Build ARIMA Models")
    analysis.get_univariate_model_results(
        params['traces'], build_arima_models_for_traces, model_args,
        specs.MODELS_COUNT, "arima", params['output_dir'])

    print("Build Linear Regression Models")
    analysis.get_univariate_model_results(
        params['traces'], build_reg_models_for_traces, model_args,
        specs.MODELS_COUNT, "reg", params['output_dir'])

    print("Build Support Vector Regression Models")
    analysis.get_univariate_model_results(
        params['traces'], build_svm_models_for_traces, model_args,
        specs.MODELS_COUNT, "svm", params['output_dir'])

    print("Build XGBoost Regression Models")
    analysis.get_univariate_model_results(
        params['traces'], build_xgb_models_for_traces, model_args,
        specs.MODELS_COUNT, "xgb", params['output_dir'])
