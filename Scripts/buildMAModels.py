"""Builds a separate moving average model for each trace and evaluates its
performance on the training and testing sets. Several types of moving average
models are built based on the different window lengths specified in
`MA_WINDOWS`.

"""
from MemoryAutoScaling import analysis, specs, utils
from MemoryAutoScaling.Models.Sequential import TraceMovingAverage


MA_WINDOWS = [1, 3, 5, 7, 10]


def build_ma_models_for_traces(trace_lst, results_dict, train_prop, max_mem):
    ma_params_lst = [{'window_length': ma_win,
                      'initial_pred': 0.0,
                      'train_prop': train_prop,
                      'max_mem': max_mem}
                     for ma_win in MA_WINDOWS]
    analysis.get_best_model_results_for_traces(
        TraceMovingAverage, ma_params_lst, trace_lst,
        results_dict, specs.MODELS_COUNT)


if __name__ == "__main__":
    analysis.run_best_models_for_all_traces(
        build_ma_models_for_traces, specs.MODELS_COUNT, "ma")
