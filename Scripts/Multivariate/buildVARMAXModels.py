"""Builds separate VARMA models for each trace and evaluates its performance
on the training and testing sets. Several types of ARIMA models are built
based on the different values of `p` and `q` specified in `VARMA_p` and
`VARMA_q`, respectively.

"""
from itertools import product
from MemoryAutoScaling import analysis, specs, utils
from MemoryAutoScaling.Models.ML import TraceVARMAX
from MemoryAutoScaling.DataHandling import MLDataHandler


VARMA_p = [p for p in range(4)]
VARMA_q = [q for q in range(4)]
param_pairs = list(product(VARMA_p, VARMA_q))
bad_val_idx = param_pairs.index((0, 0))
param_pairs = [param_pairs[i] for i in range(len(param_pairs))
               if i != bad_val_idx]
FEATURE_COLS = specs.get_lagged_trace_columns(specs.LAGS, specs.MULTI_VAR_COLS)
TARGET_COLS = specs.MULTI_VAR_COLS


def build_varmax_models_for_traces(traces_lst, results_dict, train_prop):
    data_handler = MLDataHandler(train_prop, FEATURE_COLS, TARGET_COLS)
    varmax_params_lst = [{"data_handler": data_handler, "lags": specs.LAGS,
                          "p": p, "q": q}
                          for p, q in param_pairs]
    analysis.get_best_multivariate_model_results_for_traces(
        TraceVARMAX, varmax_params_lst, traces_lst,
        results_dict, specs.MODELS_COUNT)


if __name__ == "__main__":
    analysis.run_best_multivariate_models_for_all_traces(
        build_varmax_models_for_traces, specs.MODELS_COUNT,
        "varmax", specs.MULTI_VAR_COLS)
