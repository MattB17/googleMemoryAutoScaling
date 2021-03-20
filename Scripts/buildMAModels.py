"""Builds a separate moving average model for each trace and evaluates its
performance on the training and testing sets. Several types of moving average
models are built based on the different window lengths specified in
`MA_WINDOWS`

"""

import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import product
from MemoryAutoScaling import utils, analysis
from MemoryAutoScaling.Models import MovingAverageModel
from MemoryAutoScaling.DataHandling import TraceHandler


MA_WINDOWS = [1, 3, 5, 7, 10]


def build_ma_models_for_traces(trace_lst, results_lst, train_prop):
    ma_params_lst = [{'window_length': ma_win,
                      'initial_pred': 0.0,
                      'train_prop': train_prop}
                     for ma_win in MA_WINDOWS]
    ma_models = analysis.build_models_from_params_list(
        MovingAverageModel, ma_params_lst)
    for trace in trace_lst:
        results_lst.append(
            analysis.get_model_stats_for_trace(trace, ma_models))


if __name__ == "__main__":
    traces, output_dir, train_prop = analysis.get_model_build_input_params()
    ma_results = analysis.perform_trace_modelling(
        traces, build_ma_models_for_traces, train_prop)
    ma_cols = ["{0}_mse_ma_{1}".format(mse_name, ma_win)
               for ma_win, mse_name in product(MA_WINDOWS, ["train", "test"])]
    analysis.output_model_results(
        ma_results, ["id"] + ma_cols, output_dir, "ma_results")
