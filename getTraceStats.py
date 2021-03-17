import sys
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from MemoryAutoScaling import specs
from MemoryAutoScaling import utils
from MemoryAutoScaling.Analysis import TraceAnalyzer
from MemoryAutoScaling.DataHandling import Trace, TraceHandler


CAUSAL_COLS = [specs.AVG_MEM_COL, specs.MAX_CPU_COL, specs.AVG_CPU_COL]


def run_trace_stats(traces, results_lst, causal_lags):
    analyzer = TraceAnalyzer("whitegrid", "seaborn-dark", 10, "blue", "Raw Data")
    for trace in traces:
        p_val = analyzer.test_for_stationarity(
            trace.get_maximum_memory_time_series())
        trace_df = trace.get_trace_df()
        corr_series = trace_df.corr()[specs.MAX_MEM_COL]
        trace_stats = [trace.get_trace_id(), p_val] + list(corr_series)
        for causal_col in CAUSAL_COLS:
            granger = analyzer.test_for_causality(
                trace_df, [specs.MAX_MEM_COL, causal_col], causal_lags)
            for lag in causal_lags:
                trace_stats.extend(
                    utils.get_granger_pvalues_at_lag(granger, lag))
        results_lst.append(trace_stats)


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    file_id = sys.argv[3]
    causal_lags = [lag+1 for lag in range(int(sys.argv[4]))]

    analyzer = TraceAnalyzer("whitegrid", "seaborn-dark", 10, "blue", "Raw Data")

    handler = TraceHandler(input_dir, file_id)
    handler.process_all_trace_files()
    traces = handler.get_traces()
    stat_results = mp.Manager().list()
    core_count = min(len(traces), mp.cpu_count() - 1)
    traces_per_proc = np.ceil(len(traces) / core_count)
    procs = []

    for core_num in range(core_count):
        start = int(traces_per_proc * core_num)
        end = int(min(len(traces), traces_per_proc * (core_num + 1)))
        proc = mp.Process(target=run_trace_stats,
                          args=(traces[start:end], stat_results, causal_lags))
        procs.append(proc)
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()

    stat_df = pd.DataFrame(list(stat_results))
    stat_cols = ["id", "adf_p_val"] + [
        "corr_{}".format(col_name)
        for col_name in traces[0].get_trace_df_columns()]
    stat_cols += utils.get_all_granger_col_names(CAUSAL_COLS, causal_lags)
    stat_df.columns = stat_cols
    stat_df.to_csv(
        os.path.join(output_dir, "trace_stats.csv"), sep=",", index=False)
