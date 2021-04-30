"""Builds a separate Linear Regression model for each trace and evaluates its
performance on the training and testing sets.

"""
import pandas as pd
from MemoryAutoScaling import analysis, specs, utils
from MemoryAutoScaling.Models.ML import TraceRegression
from MemoryAutoScaling.DataHandling import MLDataHandler
pd.options.mode.chained_assignment = None

FEATURE_COLS = specs.get_lagged_trace_columns(specs.LAGS)
REG_VALS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def build_reg_models_for_traces(trace_lst, results_dict,
                                train_prop, val_prop, max_mem):
    target_col = specs.get_target_variable(max_mem)
    data_handler = MLDataHandler(
        FEATURE_COLS, [target_col], train_prop, val_prop)
    reg_params_lst = [{"data_handler": data_handler,
                       'lags': specs.LAGS, 'reg_val': reg_val}
                       for reg_val in REG_VALS]
    analysis.get_best_model_results_for_traces(
        TraceRegression, reg_params_lst, trace_lst,
        results_dict, specs.MODELS_COUNT)


if __name__ == "__main__":
    analysis.run_best_models_for_all_traces(
        build_reg_models_for_traces, specs.MODELS_COUNT, "reg")
