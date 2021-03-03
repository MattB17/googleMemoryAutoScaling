"""The `ExponentialSmoothingModel` class is used to construct a prediction
model based on exponential smoothing. For exponential smoothing. The
prediction at time `t` is `p_t = alpha * x_t-1 + (1 - alpha) * p_t-1` where
`alpha` is a configurable weight constant between 0 and 1, `x_t-1` is the
actual data observation at time `t-1` and `p_t` and `p_t-1` are the
predictions at times `t` and `t-1` respectively.

"""
import numpy as np
import matplotlib.pyplot as plt
from MemoryAutoScaling import utils


class ExponentialSmoothingModel:
    """Builds an Exponential Smoothing Model.

    Parameters
    ----------
    alpha: float
        The alpha value for the model. It is a number between 0 and 1
        representing the weight of new observations versus past predictions.
    start_pred: float
        A float representing the starting prediction. This is used as the
        prediction for a new trace before seeing any data for that trace.

    Attributes
    ----------
    _alpha: float
        The alpha value for the model.
    _start_pred: float
        The starting prediction for a new, unseen trace.

    """
    def __init__(self, alpha, start_pred):
        self._alpha = alpha
        self._start_pred = start_pred

    def calculate_curr_pred(self, past_obs, past_pred):
        """Calculates the current exponential smoothing model prediction.

        Parameters
        ----------
        past_obs: float
            A float representing the past observation.
        past_pred: float
            A float representing the past prediction.

        Returns
        -------
        float
            A float representing the current exponential smoothing model
            prediction.

        """
        return (self._alpha * past_obs) + ((1 - self._alpha) * past_pred)

    def get_all_predictions(self, data_trace):
        """Calculates all predictions for `data_trace`.

        For each time point in `data_trace` the exponential smoothing
        prediction is calculated. The first prediction is `_start_pred` and
        the second prediction is the first observation as no past data exists.
        All subsequent predictions are based on the exponential smoothing
        formula.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the data trace for which the
            exponential smoothing predictions are calculated.

        Returns
        -------
        np.array
            A numpy array that is the same length as `data_trace` and has
            an exponential smoothing prediction for each time point of
            `data_trace`.

        """
        time_points = len(data_trace)
        preds = np.array([self._start_pred for _ in range(time_points)])
        if time_points >= 2:
            preds[1] = data_trace[0]
            for idx in range(2, time_points):
                preds[idx] = self.calculate_curr_pred(
                    data_trace[idx - 1], preds[idx - 1])
        return preds

    def plot_trace_and_prediction(self, data_trace):
        """Plots `data_trace` and its exponential smoothing prediction.

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
        title = "Trace vs {} Exponential Smoothing Prediction".format(
            self._alpha)
        utils.setup_trace_plot(len(data_trace), 10, title)
