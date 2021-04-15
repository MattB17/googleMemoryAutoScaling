"""Builds separate VARMA models for each trace and evaluates its performance
on the training and testing sets. Several types of ARIMA models are built
based on the different values of `p` and `q` specified in `VARMA_p` and
`VARMA_q`, respectively.

"""
from itertools import product
from MemoryAutoScaling import analysis, specs, utils
from MemoryAutoScaling.Models.Statistical import TraceVARMA

VARMA_p = [p for p in range(2)]
VARMA_q = [q for q in range(2)]
param_pairs = list(product(VARMA_p, VARMA_q))
bad_val_idx = param_pairs.index((0, 0))
param_pairs = [param_pairs[i] for i in range(len(param_pairs))
               if i != bad_val_idx]

def build_varma_models_for_traces(traces_lst, result_dict, train_prop):
    varma_params_lst = [{'model_vars': specs.MULTI_VAR_COLS,
                         'p': p, 'q': q, 'train_prop': train_prop}
                        for p, q in param_pairs]
    analysis.get_best_multivariate_model_results_for_traces(
        TraceVARMA, varma_params_lst, traces_lst,
        result_dict, specs.MODELS_COUNT)


if __name__ == "__main__":
    analysis.run_best_multivariate_models_for_all_traces(
        build_varma_models_for_traces, specs.MODELS_COUNT,
        "varma", specs.MULTI_VAR_COLS)
