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

    Attributes
    ----------
    _initial_pred: float
        Represents the initial prediction used for a new time series.

    """
    def __init__(self, initial_pred):
        self._initial_pred = initial_pred

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
    def plot_trace_and_prediction(self, data_trace):
        """Plots `data_trace` and its prediction based on the model.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the data trace being plotted.

        Returns
        -------
        None

        """
        pass

    def calculate_mse_for_trace(self, data_trace):
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
        return mean_squared_error(
            data_trace, self.get_all_predictions(data_trace))

    def _plot_trace_and_prediction(self, data_trace, title):
        """Plots `data_trace` and its prediction.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the data trace being plotted.
        title: str
            A string representing the title for the plot being produced.

        Returns
        -------
        None

        """
        plt.figure(figsize=(20, 8))
        plt.plot(data_trace, color="blue", linewidth=3)
        plt.plot(self.get_all_predictions(data_trace),
                 color="red", linewidth=3)
        tick_interval = len(data_trace) // 30
        utils.setup_trace_plot(len(data_trace), tick_interval, title)
