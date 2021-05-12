"""The `TraceModel` class is an abstract base class used for all time
series models. It represents an interface providing the basic framework used
by all time series models.

"""
from abc import ABC, abstractmethod
from MemoryAutoScaling.Evaluation import ModelResults

class TraceModel(ABC):
    """Implements the functionality of a generic trace model.

    """
    def __init__(self):
        pass

    @abstractmethod
    def get_params(self):
        """The parameters of the model.

        Returns
        -------
        dict
            A dictionary containing the parameters of the model. The keys are
            strings representing the parameter names and the corresponding
            value is the associated parameter.

        """
        pass

    @abstractmethod
    def get_model_title(self):
        """A title describing the model.

        Returns
        -------
        str
            A string representing the title for the model.

        """
        pass

    @abstractmethod
    def split_data(self, model_data, tuning=True):
        """Splits `model_data` into the training and testing set.

        If `tuning` is True, `model_data` is split into training and
        validation sets. Otherwise it is split into training + validation and
        testing sets.

        Parameters
        ----------
        model_data: pandas.Object
            A pandas Object (Series or DataFrame) containing the data used to
            build the trace model.
        tuning: bool
            A boolean value indicating whether the data is being used for
            tuning or evaluation.

        """
        pass

    @abstractmethod
    def get_model_data_for_trace(self, trace):
        """Gets the data for modeling from `trace`.

        Parameters
        ----------
        trace: Trace
            The `Trace` being modeled.

        Returns
        -------
        pandas.Object
            A pandas Object representing the data from `trace` used for
            modeling.

        """
        pass

    @abstractmethod
    def get_predictions_for_trace(self, trace, tuning=True):
        """Gets predictions for the training and testing set for `trace`.

        Parameters
        ----------
        trace: Trace
            The `Trace` for which predictions are retrieved.
        tuning: bool
            A boolean value indicating whether the model is being tuned on
            the validation set or evaluated on the test set.

        Returns
        -------
        np.array, np.array
            Two numpy arrays representing the predictions for the training and
            testing sets on `trace`.

        """
        pass

    @abstractmethod
    def fit_and_get_test_predictions(self, trace, tuning=True):
        """Fits the model and gets test predictions for `trace`.

        Parameters
        ----------
        trace: Trace
            The `Trace` for which predictions are retrieved.
        tuning: bool
            A boolean value indicating whether the model is being tuned on
            the validation set or evaluated on the test set.

        Returns
        -------
        np.array, np.array
            A numpy array representing the actual values and predictions for
            the testing set of `trace`.

        """
        pass

    @abstractmethod
    def run_model_pipeline_for_trace(self, trace, tuning=True):
        """Runs the full modeling pipeline on `trace`.

        The modeling pipeline first obtains the data needed for modeling
        from `trace`. Next it splits this data into the training and testing
        sets. It then fits the model and obtains predictions. Lastly, these
        predictions are evaluated using MASE on the training and testing sets.

        Parameters
        ----------
        trace: Trace
            The `Trace` being modeled.
        tuning: bool
            A boolean value indicating whether the model is being tuned on
            the validation set or evaluated on the test set.

        Returns
        -------
        ModelResults
            A `ModelResults` object containing the results of building the
            model on `trace`.

        """
        pass

    @abstractmethod
    def plot_trace_vs_prediction(self, trace, tuning=True):
        """Creates a plot of `trace` vs its predictions.

        Parameters
        ----------
        trace: Trace
            The `Trace` being plotted.
        tuning: bool
            A boolean value indicating whether the model is being tuned on
            the validation set or evaluated on the test set.

        Returns
        -------
        None

        """
        pass
