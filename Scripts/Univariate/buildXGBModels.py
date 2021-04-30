"""Builds a separate XGBoost Regression model for each trace and evaluates its
performance on the training and testing sets.

"""
from itertools import product
from MemoryAutoScaling import analysis, specs, utils
from MemoryAutoScaling.Models.ML import TraceXGB
from MemoryAutoScaling.DataHandling import MLDataHandler

FEATURE_COLS = specs.get_lagged_trace_columns(specs.LAGS)
LEARNING_RATES = [0.1, 0.3, 0.5]
ESTIMATORS = [10, 50, 100]
DEPTHS = [1, 2]


def build_xgb_models_for_traces(trace_lst, results_dict,
                                train_prop, val_prop, max_mem):
    target_col = specs.get_target_variable(max_mem)
    data_handler = MLDataHandler(
        FEATURE_COLS, [target_col], train_prop, val_prop)
    xgb_params_lst = [{'data_handler': data_handler,
                       'lags': specs.LAGS,
                       'learning_rate': learning_rate,
                       'estimators': n_estimators,
                       'depth': depth}
                      for learning_rate, n_estimators, depth
                      in product(LEARNING_RATES, ESTIMATORS, DEPTHS)]
    analysis.get_best_model_results_for_traces(
        TraceXGB, xgb_params_lst, trace_lst, results_dict, specs.MODELS_COUNT)


if __name__ == "__main__":
    analysis.run_best_models_for_all_traces(
        build_xgb_models_for_traces, specs.MODELS_COUNT, "xgb")
