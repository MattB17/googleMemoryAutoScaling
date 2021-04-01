"""Builds a separate exponential smoothing model for each trace and evaluates
its performance on the training and testing sets. Several types of exponential
smoothing models are built based on the different alpha values specified by
`ALPHAS`

"""
from MemoryAutoScaling import utils, analysis
from MemoryAutoScaling.Models.Sequential import TraceExponentialSmoothing


ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]


def build_es_models_for_traces(trace_lst, results_lst, train_prop):
    es_params_lst = [{'alpha': alpha_val,
                      'initial_pred': 0.001,
                      'train_prop': train_prop}
                     for alpha_val in ALPHAS]
    analysis.model_traces_and_evaluate(
        TraceExponentialSmoothing, es_params_lst, trace_lst, results_lst)


if __name__ == "__main__":
    analysis.run_models_for_all_traces(
        build_es_models_for_traces, ALPHAS, "es")
