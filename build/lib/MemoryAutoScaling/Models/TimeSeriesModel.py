"""The `TimeSeriesModel` class is an abstract base class used for all time
series models. It represents an interface providing the basic framework used
by all time series models.

"""
from abc import ABC, abstractmethod
from MemoryAutoScaling import utils
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class TimeSeriesModel(ABC):
    """Implements the functionality of a generic time series model.

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
    def __init__(self, initial_pred, train_prop=0.7):
        self._initial_pred = initial_pred
        self._train_prop = train_prop

    @abstractmethod
    def get_next_prediction(self, data_trace):
        """Calculates the next prediction for `data_trace`.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the data trace for which the prediction
            is generated.

        Returns
        -------
        float
            A float representing the prediction for the next time point based
            on `data_trace`.

        """
        pass

    @abstractmethod
    def get_all_predictions(self, data_trace):
        """Calculates all prediction for `data_trace`.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the data trace for which the
            predictions are generated.

        Returns
        -------
        np.array
            A numpy array having the same length as `data_trace` such that
            at each time point the entry in the returned array is the
            prediction for `data_trace` at that time point.

        """
        pass

    @abstractmethod
    def plot_train_trace_and_prediction(self, data_trace):
        """Plots `data_trace` and its prediction based for the training set.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the data trace being plotted.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def plot_test_trace_and_prediction(self, data_trace):
        """Plots `data_trace` and its prediction for the testing set.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the data trace being plotted.

        Returns
        -------
        None

        """
        pass

    def calculate_train_and_test_mse(self, data_trace):
        """Calculates the MSE for `data_trace` based on the model.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the actual trace from which the MSE
            will be calculated.

        Returns
        -------
        float
            A float representing the mean squared error between `data_trace`
            and its prediction based on the model.

        """
        preds = self.get_all_predictions(data_trace)
        train_cutoff = utils.get_train_cutoff(data_trace, self._train_prop)
        mse_train = mean_squared_error(
            data_trace[:train_cutoff], preds[:train_cutoff])
        mse_test = mean_squared_error(
            data_trace[train_cutoff:], preds[train_cutoff:])
        return mse_train, mse_test

    def _plot_train_trace_and_prediction(self, data_trace, title):
        """Plots the trace and its model prediction for the training set.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the actual data trace.
        title: str
            A string representing the title for the plot.

        Returns
        -------
        None

        """
        preds = self.get_all_predictions(data_trace)
        train_cutoff = utils.get_train_cutoff(data_trace, self._train_prop)
        utils.plot_trace_and_prediction(
            data_trace[:train_cutoff], preds[:train_cutoff], title)

    def _plot_test_trace_and_prediction(self, data_trace, title):
        """Plots the trace and its model prediction for the testing set.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the actual data trace.
        title: str
            A string representing the title for the plot

        Returns
        -------
        None

        """
        preds = self.get_all_predictions(data_trace)
        test_start = utils.get_train_cutoff(data_trace, self._train_prop)
        utils.plot_trace_and_prediction(
            data_trace[test_start:], preds[test_start:], title)
