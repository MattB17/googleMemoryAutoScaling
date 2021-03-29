import sys
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from MemoryAutoScaling import specs, utils, analysis
from MemoryAutoScaling.Analysis import TraceAnalyzer
from MemoryAutoScaling.DataHandling import Trace, TraceHandler


CAUSAL_COLS = [specs.AVG_MEM_COL, specs.MAX_CPU_COL, specs.AVG_CPU_COL]


def run_trace_stats(traces, results_lst, causal_lags):
    analyzer = TraceAnalyzer("whitegrid", "seaborn-dark", 10, "blue", "Raw Data")
    for trace in traces:
        p_val = analyzer.test_for_stationarity(
            trace.get_maximum_memory_time_series())
        p_val_diff = analyzer.test_for_stationarity(
            utils.get_differenced_trace(
                trace.get_maximum_memory_time_series(), 1))
        p_val_diff2 = analyzer.test_for_stationarity(
            utils.get_differenced_trace(
                trace.get_maximum_memory_time_series(), 2))
        trace_df = trace.get_trace_df()
        corr_series = trace_df.corr()[specs.MAX_MEM_COL]
        trace_stats = ([trace.get_trace_id(), p_val, p_val_diff, p_val_diff2]
                        + list(corr_series))
        for causal_col in CAUSAL_COLS:
            try:
                granger = analyzer.test_for_causality(
                    trace_df, [specs.MAX_MEM_COL, causal_col], causal_lags)
                for lag in causal_lags:
                    trace_stats.extend(
                        analysis.get_granger_pvalues_at_lag(granger, lag))
            except:
                trace_stats.extend(
                    [np.nan for _ in range(len(causal_lags) * 4)])
        results_lst.append(trace_stats)


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    file_id = sys.argv[3]
    causal_lags = [lag+1 for lag in range(int(sys.argv[4]))]
    min_trace_length = int(sys.argv[5])

    handler = TraceHandler(input_dir, file_id, min_trace_length)
    traces = handler.run_processing_pipeline()
    stat_results = mp.Manager().list()
    procs = []
    cores, traces_per_core = analysis.get_cores_and_traces_per_core(
        len(traces))

    for core_num in range(cores):
        core_traces = analysis.get_traces_for_core(
            traces, traces_per_core, core_num)
        procs.append(mp.Process(target=run_trace_stats,
                                args=(core_traces, stat_results, causal_lags)))
    analysis.initialize_and_join_processes(procs)

    stat_df = pd.DataFrame(list(stat_results))
    stat_cols = ["id", "adf_p_val", "adf_p_val_diff", "adf_p_val_diff2"] + [
        "corr_{}".format(col_name)
        for col_name in utils.get_trace_columns()]
    stat_cols += analysis.get_all_granger_col_names(CAUSAL_COLS, causal_lags)
    stat_df.columns = stat_cols
    stat_df.to_csv(
        os.path.join(output_dir, "trace_stats.csv"), sep=",", index=False)