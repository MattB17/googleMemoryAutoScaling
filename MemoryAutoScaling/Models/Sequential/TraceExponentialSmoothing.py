"""The `TraceExponentialSmoothing` class is used to construct a predictive
model based on exponential smoothing. For exponential smoothing, the
prediction at time `t` is `p_t = alpha * x_t-1 + (1 - alpha) * p_t-1` where
`alpha` is a configurable weight constant between 0 and 1, `x_t-1` is the
actual data observation at time `t-1` and `p_t` and `p_t-1` are the
predictions at times `t` and `t-1` respectively.

"""
import numpy as np
import matplotlib.pyplot as plt
from MemoryAutoScaling.Models.Sequential import SequentialModel


class TraceExponentialSmoothing(SequentialModel):
    """Builds an Exponential Smoothing model for a data trace.

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

    def _get_predictions(self, trace_ts):
        """Calculates all predictions for `trace_ts`.

        For each time point in `trace_ts` the exponential smoothing
        prediction is calculated. The first prediction is `_initial_pred` and
        the second prediction is the first observation as no past data exists.
        All subsequent predictions are based on the exponential smoothing
        formula.

        Parameters
        ----------
        trace_ts: np.array
            A numpy array representing the data trace for which the
            exponential smoothing predictions are calculated.

        Returns
        -------
        np.array, np.array
            Two numpy arrays that represent the predictions for `trace_ts`
            on the training and testing sets, respectively.

        """
        time_points = len(trace_ts)
        preds = np.array([self._initial_pred for _ in range(time_points)])
        if time_points >= 2:
            preds[1] = trace_ts[0]
            for idx in range(2, time_points):
                preds[idx] = self.get_next_prediction(
                    trace_ts[idx - 1], preds[idx - 1])
        return self.split_data(preds)

    def plot_trace_vs_prediction(self, trace):
        """Creates a plot of `trace` vs its predictions.

        The plot is arranged into two subplots. The first contains the maximum
        memory usage for the trace versus its prediction for the training set.
        The second plot is the same but for the testing set.

        Parameters
        ----------
        trace: Trace
            The `Trace` being plotted.

        Returns
        -------
        None

        """
        trace_ts = self.get_model_data_for_trace(trace)
        title = "Trace {0} vs {1}-Exponential Smoothing Prediction".format(
            trace.get_trace_id(), self._alpha)
        self._plot_time_series_vs_prediction(trace_ts, title)
