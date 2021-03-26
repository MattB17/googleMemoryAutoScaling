"""Builds separate ARIMAX models for each trace and evaluates its performance
on the training and testing sets. Several types of ARIMAX models are built
based on the different values of `p`, `d`, and `q` specified in `ARIMA_p`,
`ARIMA_d`, and `ARIMA_q`, respectively.

"""
from itertools import product
from MemoryAutoScaling import analysis, specs, utils
from MemoryAutoScaling.Models import ARIMAXModel
from MemoryAutoScaling.DataHandling import MLDataHandler


ARIMA_p = [p for p in range(8)]
ARIMA_d = [d for d in range(2)]
ARIMA_q = [q for q in range(8)]
FEATURE_COLS = [
    col for col in utils.get_lagged_trace_columns([2])
    if col != "{}_lag_2".format(specs.MAX_MEM_COL)]
TARGET_COL = specs.MAX_MEM_COL


def build_arimax_models_for_traces(traces_lst, results_lst, train_prop):
    data_handler = MLDataHandler(train_prop, FEATURE_COLS, TARGET_COL)
    arimax_params_lst = [{'data_handler': data_handler, 'lag': 2,
                          'p': p, 'd': d, 'q': q}
                          for p, d, q in product(ARIMA_p, ARIMA_d, ARIMA_q)]
    analysis.get_best_model_results_for_traces(
        ARIMAXModel, arimax_params_lst, traces_lst, results_lst, 5)


if __name__ == "__main__":
    traces, output_dir, train_prop = analysis.get_model_build_input_params()
    arimax_results = analysis.perform_trace_modelling(
        traces, build_arimax_models_for_traces, train_prop)
    arimax_cols = analysis.get_col_list_for_params(
        range(1, 6), "arimax", ["order", "train_mse", "test_mse"])
    analysis.output_model_results(
        arimax_results, ["id"] + arimax_cols, output_dir, "arimax_results")
