"""Builds a separate exponential smoothing model for each trace and evaluates
its performance on the training and testing sets. Several types of exponential
smoothing models are built based on the different alpha values specified by
`ALPHAS`

"""
import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import product
from MemoryAutoScaling import utils
from MemoryAutoScaling.Models import ExponentialSmoothingModel
from MemoryAutoScaling.DataHandling import TraceHandler


ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]


def build_es_models_for_traces(trace_lst, results_lst, train_prop):
    es_params_lst = [{'alpha': alpha_val,
                      'initial_pred': 0,
                      'train_prop': train_prop}
                     for alpha_val in ALPHAS]
    es_models = utils.build_models_from_params_list(
        ExponentialSmoothingModel, es_params_lst)
    for trace in trace_lst:
        results_lst.append(utils.get_model_stats_for_trace(trace, es_models))


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    file_id = sys.argv[3]
    min_trace_length = int(sys.argv[4])
    train_prop = float(sys.argv[5])

    trace_handler = TraceHandler(input_dir, file_id, min_trace_length)
    traces = trace_handler.run_processing_pipeline()
    es_results = mp.Manager().list()
    procs = []
    cores, traces_per_core = utils.get_cores_and_traces_per_core(len(traces))

    for core_num in range(cores):
        core_traces = utils.get_traces_for_core(
            traces, traces_per_core, core_num)
        procs.append(mp.Process(target=build_es_models_for_traces,
                                args=(core_traces, es_results, train_prop)))
    utils.initialize_and_join_processes(procs)

    es_df = pd.DataFrame(list(es_results))
    es_df.columns = ["id"] + [
        "{0}_mse_es_{1}".format(mse_name, alpha_val) for alpha_val, mse_name
        in product(ALPHAS, ["train", "test"])]
    es_df.to_csv(
        os.path.join(output_dir, "es_results.csv"), sep=",", index=False)
