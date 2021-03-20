import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import product
from MemoryAutoScaling import utils
from MemoryAutoScaling.Models import MovingAverageModel
from MemoryAutoScaling.DataHandling import TraceHandler


MA_WINDOWS = [1, 3, 5, 7, 10]


def build_ma_models_for_traces(trace_lst, results_lst, train_prop):
    ma_params_lst = [{'window_length': ma_win,
                      'initial_pred': 0.0,
                      'train_prop': train_prop}
                     for ma_win in MA_WINDOWS]
    ma_models = utils.build_models_from_params_list(
        MovingAverageModel, ma_params_lst)
    for trace in trace_lst:
        results_lst.append(utils.get_model_stats_for_trace(trace, models))


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    file_id = sys.argv[3]
    min_trace_length = int(sys.argv[4])
    train_prop = float(sys.argv[5])

    trace_handler = TraceHandler(input_dir, file_id, min_trace_length)
    traces = trace_handler.run_processing_pipeline()
    ma_results = mp.Manager().list()
    procs = []
    cores, traces_per_core = utils.get_cores_and_traces_per_core(len(traces))

    for core_num in range(cores):
        core_traces = utils.get_traces_for_core(
            traces, traces_per_core, core_num)
        procs.append(mp.Process(target=build_ma_models_for_traces,
                                args=(core_traces, ma_results, train_prop)))
    utils.initialize_and_join_processes(procs)

    ma_df = pd.DataFrame(list(ma_results))
    ma_df.columns = ["id"] + [
        "{0}_mse_ma_{1}".format(mse_name, ma_win) for ma_win, mse_name
        in product(MA_WINDOWS, ["train", "test"])]
    ma_df.to_csv(
        os.path.join(output_dir, "ma_results.csv"), sep=",", index=False)
