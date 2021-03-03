"""The `MovingAverageModel` class is used to construct a prediction model
based on a moving average.

"""
import numpy as np
import matplotlib.pyplot as plt
from MemoryAutoScaling import utils


class MovingAverageModel:
    """Builds a Moving Average Model.

    Parameters
    ----------
    window_length: int
        The window length. That is, the number of past trace observations
        used in the prediction of the current observation.
    start_pred: float
        A float representing the starting prediction. This is used as the
        prediction for a new trace before seeing any data for that trace.

    Attributes
    ----------
    _window_length: int
        The window length used in the moving average calculation.
    _start_pred: float
        The starting prediction for a new, unseen trace

    """
    def __init__(self, window_length, start_pred):
        self._window_length = window_length
        self._start_pred = start_pred

    def get_next_prediction(self, data_trace):
        """Calculates the next prediction for `data_trace`.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the data trace for which the prediction
            is calculated.

        Returns
        -------
        float
            A float representing the next prediction for `data_trace`.

        """
        if len(data_trace) < self._window_length:
            return np.average(data_trace)
        return np.average(data_trace[-1 * (self._window_length):])

    def get_all_predictions(self, data_trace):
        """Calculates all moving average prediction traces for `data_trace`.

        For each time point in `data_trace`, the moving average prediction
        is calculated. For the first time point, the prediction is
        `_start_pred` as there is no data on which the prediction can
        be based.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the data trace for which the
            predictions are calculated.

        Returns
        -------
        np.array
            A numpy array that is the same length as `data_trace` and contains
            the moving average predictions for data_trace.

        """
        preds = np.array([self._start_pred for _ in range(len(data_trace))])
        for idx in range(1, len(data_trace)):
            preds[idx] = self.get_next_prediction(data_trace[:idx])
        return preds

    def plot_trace_and_prediction(self, data_trace):
        """Plots `data_trace` and its moving average prediction.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the data trace being plotted and for
            which the predictions are calculated.

        Returns
        -------
        None

        """
        plt.figure(figsize=(20, 5))
        plt.plot(data_trace, color="blue", linewidth=3)
        plt.plot(self.get_all_predictions(data_trace),
                 color="red", linewidth=2)
        title = "Trace vs Moving Average {} Prediction".format(
            self._window_length)
        utils.setup_trace_plot(len(data_trace), 10, title)
