"""The `ModelResults` class is a container class storing the results
of training a model on a `Trace`.

"""
import numpy as np
from MemoryAutoScaling import specs, utils
from MemoryAutoScaling.Analysis.HarvestStats import HarvestStats


class ModelResults:
    """Holds model results from training a model on `Trace`.

    Parameters
    ----------
    model_params: dict
        A dictionary containing the model parameters. The keys are strings
        representing the name of the parameter and the corresponding value
        is the associated parameter.
    results_dict: dict
        A dictionary containing the modeling results.
    harvest_stats_dict: dict
        A dictionary of `HarvestStats` objects representing the harvest
        statistics calculated for the trace based on the model predictions.
        The keys are floats representing the buffer percentage and the
        corresponding key is a `HarvestStats` object representing the harvest
        stats for that trace based on the model predictions and buffer.

    Attributes
    ----------
    _model_params: dict
        The model parameters.
    _results_dict: dict
        The modeling results.
    _harvest_stats_dict: dict
        A dictionary of `HarvestStats` for the model.

    """
    def __init__(self, model_params, results_dict, harvest_stats_dict):
        self._model_params = model_params
        self._results_dict = results_dict
        self._harvest_stats_dict = harvest_stats_dict

    @classmethod
    def from_data(cls, model_params, y_train, train_preds,
                  y_test, test_preds, total_spare):
        """Builds a `ModelResults` object from training and testing data.

        Parameters
        ----------
        model_params: dict
            A dictionary containing the model parameters. The keys are strings
            representing the name of the parameter and the corresponding value
            is the associated parameter.
        y_train: np.array
            A numpy array representing actual trace values of the target
            variable for the training set.
        train_preds: np.array
            A numpy array representing model predictions for the trace on the
            training set.
        y_test: np.array
            A numpy array representing actual trace values of the target
            variable for the testing set.
        test_preds: np.array
            A numpy array representing model predictions for the trace on the
            testing set.
        total_spare: float
            A float representing the total amount of spare resource available
            for the testing period.

        Returns
        -------
        ModelResults
            The `ModelResults` object containing the results for the data.

        """
        train_preds, test_preds = utils.cap_train_and_test_predictions(
            train_preds, test_preds)
        results_dict = utils.calculate_evaluation_metrics(
            y_train, train_preds, y_test, test_preds, total_spare)
        harvest_stats_dict = {
            buffer_pct: HarvestStats.from_predictions(
                y_test, test_preds, buffer_pct)
            for buffer_pct in specs.BUFFER_PCTS}
        return cls(model_params, results_dict, harvest_stats_dict)

    @classmethod
    def build_null_model_results(cls):
        """Builds null model results.

        A null model results object is a `ModelResults` object in which all
        values are null.

        Returns
        -------
        ModelResults
            A null `ModelResults` object.

        """
        results_dict = {results_col: np.nan
                        for results_col in specs.RESULTS_COLS}
        harvest_stats_dict = {
            buffer_pct: HarvestStats.build_null_harvest_stats()
            for buffer_pct in specs.BUFFER_PCTS}
        return cls({}, results_dict, harvest_stats_dict)

    @classmethod
    def get_model_results_cols(self):
        """Gets the model results columns.

        These are the names of the variables of the `ModelResults` object,
        appearing in the same order as the list representation of the model
        results.

        Returns
        -------
        list
            A list of strings representing the names of the columns of the
            `ModelResults` object.

        """
        harvest_cols = HarvestStats.get_harvest_columns_for_buffers()
        return ["params"] + specs.RESULTS_COLS + harvest_cols


    def get_model_params(self):
        """The model parameters associated with the results.

        Returns
        -------
        tuple
            A tuple specifying the model parameters associated with the
            results.

        """
        return self._model_params

    def get_model_results(self):
        """Retrieves the model results.

        Returns
        -------
        dict
            A dictionary containing the model results.

        """
        return self._results_dict

    def to_list(self):
        """Converts the model results to a list.

        Returns
        -------
        list
            A 9-element list consisting of the model parameters and 8 floats
            representing the model results.

        """
        model_lst = [self._model_params]
        for result in specs.RESULTS_COLS:
            model_lst.append(self._results_dict[result])
        for buffer_pct in specs.BUFFER_PCTS:
            model_lst.extend(self._harvest_stats_dict[buffer_pct].to_list())
        return model_lst

    def is_better(self, other_model_results):
        """Checks if the model results are better than `other_model_results`.

        The current model results are better than `other_model_results` if
        the `HarvestStats` at the lowest buffer percentage are better than the
        `HarvestStats` at the same buffer percentage in `other_model_results`.

        Parameters
        ----------
        other_model_results: ModelResults
            The `ModelResults` object being compared to the current
            `ModelResults`.

        Returns
        -------
        bool
            True if the current `ModelResults` are better than
            `other_model_results`. Otherwise, False

        """
        lowest_buffer_pct = min(specs.BUFFER_PCTS)
        return self._harvest_stats_dict[lowest_buffer_pct].is_better(
            other_model_results._harvest_stats_dict[lowest_buffer_pct])
