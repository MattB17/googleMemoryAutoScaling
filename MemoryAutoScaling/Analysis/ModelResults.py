"""The `ModelResults` class is a container class storing the results
of training a model on a `Trace`.

"""


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
