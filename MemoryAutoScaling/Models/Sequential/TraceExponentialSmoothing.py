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
        in the training set. The default is 0.6.
    val_prop: float
        A float in the range [0, 1] representing the proportion of data in
        the validation set. The default value is 0.2.
    max_mem: bool
        A boolean indicating if the target value is maximum memory. The
        default value is True, indicating that maximum memory usage is
        the target variable. Otherwise, maximum CPU usage is used.

    Attributes
    ----------
    _alpha: float
        The alpha value for the model.
    _initial_pred: float
        The initial prediction for a new, unseen trace.
    _train_prop: float
        The proportion of data in the training set.
    _val_prop: float
        The proportion of data in the validation set.
    _max_mem: bool
        A boolean indicating the target variable. True indicates that maximum
        memory usage is the target variable. Otherwise, maximum CPU usage is
        used as the target.

    """
    def __init__(self, alpha, initial_pred, train_prop=0.6,
                 val_prop=0.2, max_mem=True):
        self._alpha = alpha
        super().__init__(initial_pred, train_prop, val_prop, max_mem)

    def get_params(self):
        """The parameters of the model.

        The model parameters are the initial prediction and the alpha value.

        Returns
        -------
        tuple
            A two element tuple containing the parameters of the model,
            corresponding to the initial prediction and the alpha value.

        """
        return {'alpha': self._alpha, 'initial_pred': self._initial_pred}

    def get_model_title(self):
        """A title describing the model.

        Returns
        -------
        str
            A string representing the title for the model.

        """
        return "{}-ExponentialSmoothing".format(self._alpha)

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

    def _get_predictions(self, trace_ts, tuning=True):
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
        tuning: bool
            A boolean value indicating whether the predictions are for the
            validation set or the test set.

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
        return self.split_data(preds, tuning)
