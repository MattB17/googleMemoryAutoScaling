"""Builds a separate ARIMA model for each trace and evaluates its performance
on the training and testing sets. Several types of ARIMA models are built
based on the different values of `p`, `d`, and `q` specified in `ARIMA_p`,
`ARIMA_d`, and `ARIMA_q` respectively.

"""
from itertools import product
from MemoryAutoScaling import analysis, utils
from MemoryAutoScaling.Models import ARIMAModel


ARIMA_p = [p for p in range(8)]
ARIMA_d = [d for d in range(2)]
ARIMA_q = [q for q in range(8)]


def build_arima_models_for_traces(traces_lst, results_lst, train_prop):
    arima_params_lst = [{'train_prop': train_prop,
                        'p': p, 'd': d, 'q': q}
                        for p, d, q in product(ARIMA_p, ARIMA_d, ARIMA_q)]
    arima_models = analysis.build_models_from_params_list(
        ARIMAModel, arima_params_lst)
    for trace in traces_lst:
        best5_results = [trace.get_trace_id()]
        for arima_model in arima_models:
            model_count = (len(best5_results) - 1) // 3
            _, train_mse, test_mse = arima_model.run_model_pipeline_for_trace(
                trace.get_maximum_memory_time_series())
            print(train_mse)
            print(test_mse)
            if (model_count < 5) or (test_mse < best5_results[-1]):
                if model_count == 0:
                    best5_results.extend([
                        arima_model.get_order(), train_mse, test_mse])
                else:
                    for idx in range(model_count):
                        if test_mse < best5_results[3 * (idx + 1)]:
                            best5_results.insert(
                                (3 * idx) + 1, arima_model.get_order())
                            best5_results.insert((3 * idx) + 2, train_mse)
                            best5_results.insert((3 * idx) + 3, test_mse)
                            break
            if len(best5_results) > 16:
                best5_results = best5_results[:16]
        results_lst.append(best5_results)


if __name__ == "__main__":
    traces, output_dir, train_prop = analysis.get_model_build_input_params()
    arima_results = analysis.perform_trace_modelling(
        traces, build_arima_models_for_traces, train_prop)
    arima_cols = ["{0}_arima_{1}".format(col, rank) for rank, col
                  in product(range(1, 6), ["order", "train_mse", "test_mse"])]
    analysis.output_model_results(
        arima_results, ["id"] + arima_cols, output_dir, "arima_results")
