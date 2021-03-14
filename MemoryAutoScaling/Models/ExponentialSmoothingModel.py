"""The `ExponentialSmoothingModel` class is used to construct a prediction
model based on exponential smoothing. For exponential smoothing. The
prediction at time `t` is `p_t = alpha * x_t-1 + (1 - alpha) * p_t-1` where
`alpha` is a configurable weight constant between 0 and 1, `x_t-1` is the
actual data observation at time `t-1` and `p_t` and `p_t-1` are the
predictions at times `t` and `t-1` respectively.

"""
import numpy as np
import matplotlib.pyplot as plt
from MemoryAutoScaling.Models import TimeSeriesModel


class ExponentialSmoothingModel(TimeSeriesModel):
    """Builds an Exponential Smoothing Model.

    Parameters
    ----------
    alpha: float
        The alpha value for the model. It is a number between 0 and 1
        representing the weight of new observations versus past predictions.
    initial_pred: float
        A float representing the initial prediction. This is used as the
        prediction for a new trace before seeing any data for that trace.
    train_prop: float
        A float in the range [0, 1], representing the proportion of data
        in the training set. The default is 0.7.

    Attributes
    ----------
    _alpha: float
        The alpha value for the model.
    _initial_pred: float
        The initial prediction for a new, unseen trace.
    _train_prop: float
        The proportion of data in the training set.

    """
    def __init__(self, alpha, initial_pred, train_prop=0.7):
        self._alpha = alpha
        super().__init__(initial_pred, train_prop)

    def get_next_prediction(self, past_obs, past_pred):
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
        preds = np.array([self._initial_pred for _ in range(time_points)])
        if time_points >= 2:
            preds[1] = data_trace[0]
            for idx in range(2, time_points):
                preds[idx] = self.get_next_prediction(
                    data_trace[idx - 1], preds[idx - 1])
        return preds

    def plot_train_trace_and_prediction(self, data_trace):
        """Plots `data_trace` and its prediction for the training set.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the data trace being plotted and for
            which the predictions are calculated.

        Returns
        -------
        None

        """
        title = "Trace vs {} Exponential Smoothing Prediction - Train".format(
            self._alpha)
        super()._plot_train_trace_and_prediction(data_trace, title)

    def plot_test_trace_and_prediction(self, data_trace):
        """Plots `data_trace` and its prediction for the testing set.

        Parameters
        ----------
        data_trace: np.array
            A numpy array representing the data trace being plotted and for
            which the predictions are calculated.

        Returns
        -------
        None

        """
        title = "Trace vs {} Exponential Smoothing Prediction - Test".format(
            self._alpha)
        super()._plot_test_trace_and_prediction(data_trace, title)
