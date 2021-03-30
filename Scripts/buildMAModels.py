"""Builds a separate moving average model for each trace and evaluates its
performance on the training and testing sets. Several types of moving average
models are built based on the different window lengths specified in
`MA_WINDOWS`

"""
from MemoryAutoScaling import utils, analysis
from MemoryAutoScaling.Models.Sequential import TraceMovingAverage


MA_WINDOWS = [1, 3, 5, 7, 10]


def build_ma_models_for_traces(trace_lst, results_lst, train_prop):
    ma_params_lst = [{'window_length': ma_win,
                      'initial_pred': 0.0,
                      'train_prop': train_prop}
                     for ma_win in MA_WINDOWS]
    analysis.model_traces_and_evaluate(
        TraceMovingAverage, ma_params_lst, trace_lst, results_lst)


if __name__ == "__main__":
    traces, output_dir, train_prop = analysis.get_model_build_input_params()
    ma_results = analysis.perform_trace_modelling(
        traces, build_ma_models_for_traces, train_prop)
    ma_cols = analysis.get_col_list_for_params(
        MA_WINDOWS, "ma", ["train_mse", "test_mse"])
    analysis.output_model_results(
        ma_results, ["id"] + ma_cols, output_dir, "ma_results")
