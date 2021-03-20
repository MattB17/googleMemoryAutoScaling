"""Builds a separate Linear Regression model for each trace and evaluates its
performance on the training and testing sets.

"""
import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from MemoryAutoScaling import specs, utils, analysis
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
    traces, output_dir, train_prop = analysis.get_model_build_input_params()
    reg_results = analysis.perform_trace_modelling(
        traces, build_reg_models_for_traces, train_prop)
    reg_cols = ["id", "train_mse", "test_mse"]
    analysis.output_model_results(
        reg_results, reg_cols, output_dir, "reg_results")
