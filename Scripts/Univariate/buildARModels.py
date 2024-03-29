"""Builds a separate auto-regressive model for each trace and evaluates its
performance on the training and testing sets. Several types of auto-regressive
models are built based on the different auto regression parameters specified
in `AR_PARAMS`.

"""
import pandas as pd
from MemoryAutoScaling import analysis, specs, utils
from MemoryAutoScaling.Models.Statistical import TraceAR
pd.options.mode.chained_assignment = None


AR_PARAMS = [1, 2, 3, 4, 5, 7, 10]


def build_ar_models_for_traces(trace_lst, results_dict, train_prop,
                               val_prop, max_mem):
    fixed_model_params = {'train_prop': train_prop,
                          'val_prop': val_prop,
                          'max_mem': max_mem}
    ar_params_lst = [{'p': p, **fixed_model_params} for p in AR_PARAMS]
    analysis.get_best_model_results_for_traces(
        TraceAR, ar_params_lst, trace_lst,
        results_dict, specs.MODELS_COUNT, fixed_model_params)


if __name__ == "__main__":
    analysis.run_best_models_for_all_traces(
        build_ar_models_for_traces, specs.MODELS_COUNT, "ar")
