"""Builds separate ARIMA models for each trace and evaluates its performance
on the training and testing sets. Several types of ARIMA models are built
based on the different values of `p`, `d`, and `q` specified in `ARIMA_p`,
`ARIMA_d`, and `ARIMA_q`, respectively.

"""
from itertools import product
from MemoryAutoScaling import analysis, utils
from MemoryAutoScaling.Models import ARIMAModel


ARIMA_p = [p for p in range(8)]
ARIMA_d = [d for d in range(2)]
ARIMA_q = [q for q in range(8)]


def build_arima_models_for_traces(traces_lst, results_lst, train_prop):
    arima_params_lst = [{'train_prop': train_prop,
                        'p': p, 'd': d, 'q': q}
                        for p, d, q in product(ARIMA_p, ARIMA_d, ARIMA_q)]
    analysis.get_best_model_results_for_traces(
        ARIMAModel, arima_params_lst, traces_lst, results_lst, 5)


if __name__ == "__main__":
    traces, output_dir, train_prop = analysis.get_model_build_input_params()
    arima_results = analysis.perform_trace_modelling(
        traces, build_arima_models_for_traces, train_prop)
    arima_cols = analysis.get_col_list_for_params(
        range(1, 6), "arima", ["order", "train_mse", "test_mse"])
    analysis.output_model_results(
        arima_results, ["id"] + arima_cols, output_dir, "arima_results")