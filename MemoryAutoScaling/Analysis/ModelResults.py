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
    def __init__(self, model_params, y_train,
                 train_preds, y_test, test_preds):
        self._model_params = model_params
        self._results_dict = utils.calculate_evaluation_metrics(
            y_train, train_preds, y_test, test_preds)

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
