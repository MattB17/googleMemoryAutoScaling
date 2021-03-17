import sys
import numpy as np
import pandas as pd
from MemoryAutoScaling import specs
from MemoryAutoScaling import utils
from MemoryAutoScaling.Analysis import TraceAnalyzer
from MemoryAutoScaling.DataHandling import Trace, TraceHandler


CAUSAL_COLS = [specs.AVG_MEM_COL, specs.MAX_CPU_COL, specs.AVG_CPU_COL]


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    file_id = sys.argv[3]
    causal_lags = [lag+1 for lag in range(int(sys.argv[4]))]

    analyzer = TraceAnalyzer("whitegrid", "seaborn-dark", 10, "blue", "Raw Data")

    handler = TraceHandler(input_dir, file_id)
    handler.process_all_trace_files()

    stat_results = []
    for trace in handler.get_traces():
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
        stat_results.append(trace_stats)
    stat_df = pd.DataFrame(stat_results)
    stat_cols = ["ID", "ADF p-val"] + list(corr_series.index.values)
    stat_cols += utils.get_all_granger_col_names(CAUSAL_COLS, causal_lags)
    stat_df.columns = stat_cols
    print(stat_df)
