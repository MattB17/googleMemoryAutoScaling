"""Builds a separate Linear Regression model for each trace and evaluates its
performance on the training and testing sets.

"""
from MemoryAutoScaling import analysis, specs, utils
from MemoryAutoScaling.Models import TimeSeriesRegression
from MemoryAutoScaling.DataHandling import MLDataHandler


FEATURE_COLS = utils.get_lagged_trace_columns([2])
TARGET_COL = specs.MAX_MEM_COL


def build_reg_models_for_traces(trace_lst, results_lst, train_prop):
    data_handler = MLDataHandler(train_prop, FEATURE_COLS, TARGET_COL)
    reg_model = TimeSeriesRegression(data_handler, [2])
    for trace in trace_lst:
        _, train_mse, test_mse = reg_model.run_model_pipeline_for_trace(
            trace)
        results_lst.append([trace.get_trace_id(), train_mse, test_mse])


if __name__ == "__main__":
    traces, output_dir, train_prop = analysis.get_model_build_input_params()
    reg_results = analysis.perform_trace_modelling(
        traces, build_reg_models_for_traces, train_prop)
    reg_cols = ["id", "train_mse", "test_mse"]
    analysis.output_model_results(
        reg_results, reg_cols, output_dir, "reg_results")
