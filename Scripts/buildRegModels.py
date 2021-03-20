"""Builds a separate Linear Regression model for each trace and evaluates its
performance on the training and testing sets.

"""
import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from MemoryAutoScaling import specs
from MemoryAutoScaling import utils
from MemoryAutoScaling.Models import TimeSeriesRegression
from MemoryAutoScaling.DataHandling import MLDataHandler, TraceHandler


FEATURE_COLS = utils.get_lagged_trace_columns()
TARGET_COL = specs.MAX_MEM_COL


def build_reg_models_for_traces(trace_lst, results_lst, train_prop):
    data_handler = MLDataHandler(train_prop, FEATURE_COLS, TARGET_COL)
    reg_model = TimeSeriesRegression(data_handler)
    for trace in trace_lst:
        _, train_mse, test_mse = reg_model.run_model_pipeline_on_raw_data(
            trace.get_lagged_df(1))
        results_lst.append([trace.get_trace_id(), train_mse, test_mse])


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    file_id = sys.argv[3]
    min_trace_length = int(sys.argv[4])
    train_prop = float(sys.argv[5])

    trace_handler = TraceHandler(input_dir, file_id, min_trace_length)
    traces = trace_handler.run_processing_pipeline()
    reg_results = mp.Manager().list()
    procs = []
    cores, traces_per_core = utils.get_cores_and_traces_per_core(len(traces))

    for core_num in range(cores):
        core_traces = utils.get_traces_for_core(
            traces, traces_per_core, core_num)
        procs.append(mp.Process(target=build_reg_models_for_traces,
                                args=(core_traces, reg_results, train_prop)))
    utils.initialize_and_join_processes(procs)


    reg_df = pd.DataFrame(list(reg_results))
    reg_df.columns = ["id", "train_mse", "test_mse"]
    reg_df.to_csv(
        os.path.join(output_dir, "reg_results.csv"), sep=",", index=False)
