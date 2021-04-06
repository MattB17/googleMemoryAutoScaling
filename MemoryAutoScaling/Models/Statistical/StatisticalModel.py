"""The `StatisticalModel` class is an abstract base class used for all
statistical models used for time series prediction. It represents the
interface used by these models.

"""
from abc import abstractmethod
from MemoryAutoScaling import utils
from MemoryAutoScaling.Models import TraceModel


class StatisticalModel(TraceModel):
    """A statistical model for a data trace.

    Parameters
    ----------
    train_prop: float
        A float in the range [0, 1] representing the proportion of data
        in the training set. The default value is 0.7.

    Attributes
    ----------
    _train_prop: float
        The proportion of data in the training set.
    _model: Object
        The underlying model being fit to the trace data.

    """
    def __init__(self, train_prop=0.7):
        self._train_prop = train_prop
        self._model = None

    def get_predictions_for_trace(self, trace):
        """Gets predictions for the training and testing sets for `trace`.

        Parameters
        ----------
        trace: Trace
            The `Trace` for which predictions are retrieved.

        Returns
        -------
        pd.Object, pd.Object
            Two pandas objects representing the predictions for the training
            and testing sets of `trace`, respectively.

        """
        trace_data = self.get_model_data_for_trace
        train_data, test_data = self.split_data(trace_data)
        return self._get_predictions(len(test_data))

    def get_plot_title(self, trace):
        """Gets the plot title based on `trace`.

        Parameters
        ----------
        trace: Trace
            The `Trace` object for which the plot title is generated.

        Returns
        -------
        str
            A string representing the plot title for `trace`.

        """
        return "Trace {0} {1} Model Actual vs. Predicted".format(
            trace.get_trace_id(), self.get_model_title())

    @abstractmethod
    def _fit(self, train_data):
        """Fits the model based on `train_data`.

        Parameters
        ----------
        train_data: pd.Object
            A pandas Object representing the training data used to fit the
            model.

        Returns
        -------
        None

        """
        pass

    def _get_predictions(self, forecast_length):
        """Retrieves the predictions for `forecast_length`.

        The predictions are retrieved for the training period plus the period
        ending `forecast_length` time points after the training period.

        Parameters
        ----------
        forecast_len: int
            An integer representing the length of the forecast interval.

        Returns
        -------
        pd.Object, pd.Object
            Two pandas Objects representing the predictions for the training
            period and predictions for the period of `forecast_length` time
            intervals after the training period, respectively.

        """
        preds = self._model.get_prediction(
            end=self._model.nobs + forecast_length - 1)
        mean_preds = utils.impute_for_time_series(preds.predicted_mean, 0)
        return mean_preds[:-forecast_length], mean_preds[-forecast_length:]
