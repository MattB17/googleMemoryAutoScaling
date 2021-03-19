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
    print("Processing Traces")
    print("-----------------")
    trace_handler.process_all_trace_files()
    print("Processing Complete")
    print()
    traces = trace_handler.get_traces()
    reg_results = mp.Manager().list()
    core_count = min(len(traces), mp.cpu_count() - 1)
    traces_per_proc = np.ceil(len(traces) / core_count)
    procs = []

    for core_num in range(core_count):
        start = int(traces_per_proc * core_num)
        end = int(min(len(traces), traces_per_proc * (core_num + 1)))
        proc = mp.Process(target=build_reg_models_for_traces,
                          args=(traces[start:end], reg_results, train_prop))
        procs.append(proc)
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()

    reg_df = pd.DataFrame(list(reg_results))
    reg_df.columns = ["id", "train_mse", "test_mse"]
    reg_df.to_csv(
        os.path.join(output_dir, "reg_results.csv"), sep=",", index=False)
