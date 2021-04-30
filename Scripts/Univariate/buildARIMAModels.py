"""Builds separate ARIMA models for each trace and evaluates its performance
on the training and testing sets. Several types of ARIMA models are built
based on the different values of `p`, `d`, and `q` specified in `ARIMA_p`,
`ARIMA_d`, and `ARIMA_q`, respectively.

"""
from itertools import product
from MemoryAutoScaling import analysis, specs, utils
from MemoryAutoScaling.Models.Statistical import TraceARIMA


ARIMA_p = [p for p in range(5)]
ARIMA_d = [d for d in range(2)]
ARIMA_q = [q for q in range(5)]

MA_PARAMS = [(0, 0, q) for q in ARIMA_q[1:]]
AR_PARAMS = [(p, 0, 0) for p in ARIMA_p[1:]]
ARIMA_PARAMS = [arima_tup for arima_tup in product(ARIMA_p, ARIMA_d, ARIMA_q)
                if arima_tup not in MA_PARAMS + AR_PARAMS + [(0, 0, 0)]]


def build_arima_models_for_traces(traces_lst, results_dict,
                                  train_prop, val_prop, max_mem):
    arima_params_lst = [{'train_prop': train_prop,
                         'val_prop': val_prop,
                        'p': p, 'd': d, 'q': q,
                        'max_mem': max_mem}
                        for p, d, q in ARIMA_PARAMS]
    analysis.get_best_model_results_for_traces(
        TraceARIMA, arima_params_lst, traces_lst,
        results_dict, specs.MODELS_COUNT)


if __name__ == "__main__":
    analysis.run_best_models_for_all_traces(
        build_arima_models_for_traces, specs.MODELS_COUNT, "arima")
