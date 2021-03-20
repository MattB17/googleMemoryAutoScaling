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
from MemoryAutoScaling import utils, analysis
from MemoryAutoScaling.Models import ExponentialSmoothingModel
from MemoryAutoScaling.DataHandling import TraceHandler


ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]


def build_es_models_for_traces(trace_lst, results_lst, train_prop):
    es_params_lst = [{'alpha': alpha_val,
                      'initial_pred': 0,
                      'train_prop': train_prop}
                     for alpha_val in ALPHAS]
    es_models = analysis.build_models_from_params_list(
        ExponentialSmoothingModel, es_params_lst)
    for trace in trace_lst:
        results_lst.append(
            analysis.get_model_stats_for_trace(trace, es_models))


if __name__ == "__main__":
    traces, output_dir, train_prop = analysis.get_model_build_input_params()
    es_results = analysis.perform_trace_modelling(
        traces, build_es_models_for_traces, train_prop)
    es_cols = ["{0}_mse_es_{1}".format(mse_name, alpha_val)
               for alpha_val, mse_name in product(ALPHAS, ["train", "test"])]
    analysis.output_model_results(
        es_results, ["id"] + es_cols, output_dir, "es_results")
