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
    analysis.run_models_for_all_traces(
        build_ma_models_for_traces, MA_WINDOWS, "ma")
