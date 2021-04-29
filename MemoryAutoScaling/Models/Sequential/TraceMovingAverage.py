"""The `MovingAverageModel` class is used to construct a prediction model
based on a moving average.

"""
import numpy as np
import matplotlib.pyplot as plt
from MemoryAutoScaling.Models.Sequential import SequentialModel


class TraceMovingAverage(SequentialModel):
    """Builds a Moving Average Model for a data trace.

    Parameters
    ----------
    window_length: int
        The window length. That is, the number of past trace observations
        used in the prediction of the current observation.
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
    _window_length: int
        The window length used in the moving average calculation.
    _initial_pred: float
        The initial prediction for a new, unseen trace
    _train_prop: float
        The proportion of data in the training set.
    _val_prop: float
        The proportion of data in the validation set.
    _max_mem: bool
        A boolean indicating the target variable. True indicates that maximum
        memory usage is the target variable. Otherwise, maximum CPU usage is
        used as the target.

    """
    def __init__(self, window_length, initial_pred,
                 train_prop=0.6, val_prop=0.2, max_mem=True):
        self._window_length = window_length
        super().__init__(initial_pred, train_prop, val_prop, max_mem)

    def get_params(self):
        """The parameters of the model.

        The model parameters are the initial prediction and window length.

        Returns
        -------
        tuple
            A 2 element tuple containing the model parameters, corresponding
            to the initial prediction and the window length.

        """
        return {'window_length': self._window_length,
                'initial_pred': self._initial_pred}

    def get_model_title(self):
        """A title describing the model.

        Returns
        -------
        str
            A string representing the title for the model.

        """
        return "{}-MovingAverage".format(self._window_length)

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

    def _get_predictions(self, trace_ts, tuning=True):
        """Calculates all moving average prediction traces for `trace_ts`.

        For each time point in `trace_ts`, the moving average prediction
        is calculated. For the first time point, the prediction is
        `_initial_pred` as there is no data on which the prediction can
        be based.

        Parameters
        ----------
        trace_ts: np.array
            A numpy array representing the data trace for which the
            predictions are calculated.
        tuning: bool
            A boolean value indicating whether the predictions are for the
            validation set or the test set.

        Returns
        -------
        np.array, np.array
            Two numpy arrays that represent the predictions for `trace_ts`
            on the training and testing sets, respectively.

        """
        preds = np.array([self._initial_pred for _ in range(len(trace_ts))])
        for idx in range(1, len(trace_ts)):
            preds[idx] = self.get_next_prediction(trace_ts[:idx])
        return self.split_data(preds, tuning)
