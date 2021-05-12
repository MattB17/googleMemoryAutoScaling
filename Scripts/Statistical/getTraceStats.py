import sys
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from MemoryAutoScaling import analysis, parallel, specs, utils
from MemoryAutoScaling.Evaluation import TraceAnalyzer
from MemoryAutoScaling.DataHandling import Trace, TraceHandler


def run_trace_stats(traces, results_lst, causal_lags,
                    causal_cols, output_target):
    analyzer = TraceAnalyzer("whitegrid", "seaborn-dark", 10, "blue", "Raw Data")
    trace_count = len(traces)
    for idx in range(trace_count):
        trace = traces[idx]
        trace_ts = trace.get_target_time_series(output_target)
        utilization = trace.get_resource_utilization(output_target)
        allocated = trace.get_total_allocated_resource(output_target)
        p_val = analyzer.test_for_stationarity(trace_ts)
        p_val_diff = analyzer.test_for_stationarity(
            utils.get_differenced_trace(trace_ts, 1))
        p_val_diff2 = analyzer.test_for_stationarity(
            utils.get_differenced_trace(trace_ts, 2))
        trace_df = trace.get_lagged_df(specs.LAGS)
        corr_series = trace_df.corr()[output_target]
        trace_stats = ([trace.get_trace_id(), utilization, allocated, p_val,
                        p_val_diff, p_val_diff2] + list(corr_series))
        for causal_col in causal_cols:
            try:
                granger = analyzer.test_for_causality(
                    trace_df, [output_target, causal_col], specs.LAGS)
                for lag in specs.LAGS:
                    trace_stats.extend(
                        analysis.get_granger_pvalues_at_lag(granger, lag))
            except:
                trace_stats.extend(
                    [np.nan for _ in range(len(causal_lags) * 4)])
        results_lst.append(trace_stats)
        analysis.log_modeling_progress(idx, trace_count)



if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    file_id = sys.argv[3]
    min_trace_length = int(sys.argv[4])
    aggregation_window = int(sys.argv[5])
    max_mem = (sys.argv[6].lower() == "true")

    handler = TraceHandler(
        input_dir, file_id, min_trace_length, aggregation_window)
    traces = handler.run_processing_pipeline()
    stat_results = mp.Manager().list()
    procs = []
    cores, traces_per_core = parallel.get_cores_and_traces_per_core(
        len(traces))

    input_target, output_target = specs.get_input_and_output_target(max_mem)
    causal_cols = specs.get_causal_cols(input_target)

    for core_num in range(cores):
        core_traces = parallel.get_traces_for_core(
            traces, traces_per_core, core_num)
        stats_args = (core_traces, stat_results, specs.LAGS,
                      causal_cols, output_target)
        procs.append(mp.Process(target=run_trace_stats, args=stats_args))
    parallel.initialize_and_join_processes(procs)

    stat_df = pd.DataFrame(list(stat_results))
    stat_cols = ["id", "utilization", "allocated", "adf_p_val",
                 "adf_p_val_diff", "adf_p_val_diff2"]
    stat_cols += ["corr_{}".format(col_name) for col_name in
                  specs.get_trace_columns() +
                  specs.get_lagged_trace_columns(specs.LAGS)]
    stat_cols += analysis.get_all_granger_col_names(causal_cols, specs.LAGS)
    stat_df.columns = stat_cols
    stat_df.to_csv(
        os.path.join(output_dir, "trace_stats.csv"), sep=",", index=False)
