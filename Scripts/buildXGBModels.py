"""Builds a separate XGBoost Regression model for each trace and evaluates its
performance on the training and testing sets.

"""
from MemoryAutoScaling import analysis, specs, utils
from MemoryAutoScaling.Models import TimeSeriesXGB
from MemoryAutoScaling.DataHandling import MLDataHandler

LAGS = [2, 3, 4]
FEATURE_COLS = utils.get_lagged_trace_columns(LAGS)
TARGET_COL = specs.MAX_MEM_COL


def build_xgb_models_for_traces(trace_lst, results_lst, train_prop):
    data_handler = MLDataHandler(train_prop, FEATURE_COLS, TARGET_COL)
    xgb_model = TimeSeriesXGB(data_handler, LAGS, 0.1, 500, 2)
    for trace in trace_lst:
        _, train_mse, test_mse = xgb_model.run_model_pipeline_for_trace(trace)
        results_lst.append([trace.get_trace_id(), train_mse, test_mse])


if __name__ == "__main__":
    traces, output_dir, train_prop = analysis.get_model_build_input_params()
    xgb_results = analysis.perform_trace_modelling(
        traces, build_xgb_models_for_traces, train_prop)
    xgb_cols = ["id", "train_mse", "test_mse"]
    analysis.output_model_results(
        xgb_results, xgb_cols, output_dir, "xgb_results")
