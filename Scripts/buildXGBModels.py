"""Builds a separate XGBoost Regression model for each trace and evaluates its
performance on the training and testing sets.

"""
from itertools import product
from MemoryAutoScaling import analysis, specs, utils
from MemoryAutoScaling.Models import TimeSeriesXGB
from MemoryAutoScaling.DataHandling import MLDataHandler

LAGS = [2, 3, 4]
FEATURE_COLS = utils.get_lagged_trace_columns(LAGS)
TARGET_COL = specs.MAX_MEM_COL
LEARNING_RATES = [0.01, 0.03, 0.1, 0.3, 0.5]
ESTIMATORS = [10, 50, 100, 200, 500, 1000]
DEPTHS = [1, 2, 5]


def build_xgb_models_for_traces(trace_lst, results_lst, train_prop):
    data_handler = MLDataHandler(train_prop, FEATURE_COLS, TARGET_COL)
    xgb_params_lst = [{'data_handler': data_handler,
                       'lags': LAGS,
                       'learning_rate': learning_rate,
                       'estimators': n_estimators,
                       'depth': depth}
                      for learning_rate, n_estimators, depth
                      in product(LEARNING_RATES, ESTIMATORS, DEPTHS)]
    analysis.get_best_model_results_for_traces(
        TimeSeriesXGB, xgb_params_lst, trace_lst, results_lst, 5)


if __name__ == "__main__":
    traces, output_dir, train_prop = analysis.get_model_build_input_params()
    xgb_results = analysis.perform_trace_modelling(
        traces, build_xgb_models_for_traces, train_prop)
    xgb_cols = analysis.get_col_list_for_params(
        range(1, 6), "xgb", ["params", "train_mse", "test_mse"])
    analysis.output_model_results(
        xgb_results, ["id"] + xgb_cols, output_dir, "xgb_results")
