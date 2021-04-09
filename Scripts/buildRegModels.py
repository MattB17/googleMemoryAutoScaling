"""Builds a separate Linear Regression model for each trace and evaluates its
performance on the training and testing sets.

"""
from MemoryAutoScaling import analysis, specs, utils
from MemoryAutoScaling.Models.ML import TraceRegression
from MemoryAutoScaling.DataHandling import MLDataHandler

LAGS = [2, 3, 4]
FEATURE_COLS = specs.get_lagged_trace_columns(LAGS)
TARGET_COL = specs.MAX_MEM_COL
REG_VALS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def build_reg_models_for_traces(trace_lst, results_lst, train_prop):
    data_handler = MLDataHandler(train_prop, FEATURE_COLS, TARGET_COL)
    reg_params_lst = [{"data_handler": data_handler,
                       'lags': LAGS, 'reg_val': reg_val}
                       for reg_val in REG_VALS]
    analysis.model_traces_and_evaluate(
        TraceRegression, reg_params_lst, trace_lst, results_lst)


if __name__ == "__main__":
    analysis.run_models_for_all_traces(
        build_reg_models_for_traces, REG_VALS, "reg")
