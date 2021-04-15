"""The `ModelResults` class is a container class storing the results
of training a model on a `Trace`.

"""
from MemoryAutoScaling import specs, utils


class ModelResults:
    """Holds model results from training a model on `Trace`.

    Parameters
    ----------
    model_params: tuple
        A tuple containing the model parameters.
    results_dict: dict
        A dictionary containing the modeling results.

    Attributes
    ----------
    _model_params: tuple
        The model parameters.
    _results_dict: dict
        The modeling results.

    """
    def __init__(self, model_params, results_dict):
        self._model_params = model_params
        self._results_dict = results_dict

    @classmethod
    def from_data(cls, model_params, y_train,
                  train_preds, y_test, test_preds):
        """Builds a `ModelResults` object from training and testing data.

        Parameters
        ----------
        model_params: tuple
            A tuple representing the parameters of the model fit to the data.
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

        Returns
        -------
        ModelResults
            The `ModelResults` object containing the results for the data.

        """
        results_dict = utils.calculate_evaluation_metrics(
            y_train, train_preds, y_test, test_preds)
        return cls(model_params, results_dict)

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
        return model_lst

    def is_better(self, other_model_results):
        """Checks if the model results are better than `other_model_results`.

        The current model results are better than `other_model_results` if
        the weighted mean absolute scaled error is lower. The weigthed mean
        absolute scaled error is the total mean absolute scaled error plus
        `w` times the one-sided mean absolute scaled error of under
        predictions, where `w` is `specs.OVERALL_MASE_WEIGHT`.

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
        w = specs.OVERALL_MASE_WEIGHT
        curr_mase = (self.get_model_results()['test_mase'] +
                     (w * self.get_model_results()['under_mase']))
        other_mase = \
            (other_model_results.get_model_results()['test_mase'] +
             (w * other_model_results.get_model_results()['under_mase']))
        return curr_mase < other_mase
