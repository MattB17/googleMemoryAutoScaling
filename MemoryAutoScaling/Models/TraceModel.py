"""The `TraceModel` class is an abstract base class used for all time
series models. It represents an interface providing the basic framework used
by all time series models.

"""
from abc import ABC, abstractmethod
from MemoryAutoScaling import utils
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

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
        tuple
            A tuple containing the parameters of the model.

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
    def split_data(self, model_data):
        """Splits `model_data` into the training and testing set.

        Parameters
        ----------
        model_data: pandas.Object
            A pandas Object (Series or DataFrame) containing the data used to
            build the trace model.

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
    def get_predictions_for_trace(self, trace):
        """Gets predictions for the training and testing set for `trace`.

        Parameters
        ----------
        trace: Trace
            The `Trace` for which predictions are retrieved.

        Returns
        -------
        np.array, np.array
            Two numpy arrays representing the predictions for the training and
            testing sets on `trace`.

        """
        pass

    @abstractmethod
    def run_model_pipeline_for_trace(self, trace):
        """Runs the full modeling pipeline on `trace`.

        The modeling pipeline first obtains the data needed for modeling
        from `trace`. Next it splits this data into the training and testing
        sets. It then fits the model and obtains predictions. Lastly, these
        predictions are evaluated using MAPE on the training and testing sets.

        Parameters
        ----------
        trace: Trace
            The `Trace` being modeled.

        Returns
        -------
        tuple
            A tuple of six floats. The first two represent the mean absolute
            percentage error for the training and testing sets, respectively.
            The next two represent the proportion of under predictions and the
            magnitude of the maximum under prediction, respectively. The last
            two represent the proportion of over predictions and the magnitude
            of the average over prediction.

        """
        pass

    @abstractmethod
    def plot_trace_vs_prediction(self, trace):
        """Creates a plot of `trace` vs its predictions.

        Parameters
        ----------
        trace: Trace
            The `Trace` being plotted.

        Returns
        -------
        None

        """
        pass
