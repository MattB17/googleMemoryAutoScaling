"""Builds a separate exponential smoothing model for each trace and evaluates
its performance on the training and testing sets. Several types of exponential
smoothing models are built based on the different alpha values specified by
`ALPHAS`

"""
from MemoryAutoScaling import analysis, specs, utils
from MemoryAutoScaling.Models.Sequential import TraceExponentialSmoothing


ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]


def build_es_models_for_traces(trace_lst, results_dict,
                               train_prop, val_prop, max_mem):
    es_params_lst = [{'alpha': alpha_val,
                      'initial_pred': 0.001,
                      'train_prop': train_prop,
                      'val_prop': val_prop,
                      'max_mem': max_mem}
                     for alpha_val in ALPHAS]
    analysis.get_best_model_results_for_traces(
        TraceExponentialSmoothing, es_params_lst, trace_lst,
        results_dict, specs.MODELS_COUNT)


if __name__ == "__main__":
    analysis.run_best_models_for_all_traces(
        build_es_models_for_traces, specs.MODELS_COUNT, "es")
