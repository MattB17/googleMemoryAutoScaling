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

    Parameters
    ----------
    initial_pred: float
        A float representing an initial prediction used for a new time series.
    train_prop: float
        A float in the range [0, 1] representing the proportion of
        observations in the training set. The default value is 0.7.

    Attributes
    ----------
    _initial_pred: float
        Represents the initial prediction used for a new time series.
    _train_prop: float
        Represents the percent of data in the training set.

    """
    def __init__(self, train_prop=0.7):
        self._train_prop = train_prop

    @abstractmethod
    def split_data(self, model_data):
        """Splits `model_data` into the training and testing set.

        Parameters
        ----------
        model_data: pandas.Object
            A pandas Object (Series or DataFrame) containing the data used to
            build the trace model.

        Returns
        -------
        pandas.Object, pandas.Object
            The two pandas objects obtained from `model_data` after applying
            the train-test split.

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
        predictions are evaluated using MSE on the training and testing sets.

        Parameters
        ----------
        trace: Trace
            The `Trace` being modeled.

        Returns
        -------
        float, float
            Two floats representing the MSEs for the training and testing sets
            after modeling `trace`.

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
