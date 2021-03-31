"""The `TraceSVM` class builds a Support Vector Machine model to predict
future time points for a time series.

"""
from MemoryAutoScaling.Models.ML import MLModel
from sklearn.svm import SVR


class TraceSVM(MLModel):
    """A Support Vector Machine model for time series data.

    Parameters
    ----------
    data_handler: MLDataHandler
        An `MLDataHandler` used to pre-process data for the model.
    lags: list
        A list of integers representing the lags to use for time series
        features to the model. That is, for any time series used as a feature
        to the model, the model will use the value of the time series lagged
        by the time points in `lags` when predicting for the target variable.
    reg_val: float
        A float specifying the degree of regularization for the model.

    Attributes
    ----------
    _model_name: str
        The name of the machine learning model.
    _data_handler: MLDataHandler
        The handler used to pre-process data for the model.
    _lags: list
        The lags used for time series features to the model.
    _model: Object
        The underlying machine learning model being fit.
    _is_fit: bool
        Indicates if the model has been fit to training data.
    _reg_val: float
        The regularization parameter for the model.

    """
    def __init__(self, data_handler, lags, reg_val):
        super().__init__("TimeSeriesSVM", data_handler, lags)
        self._reg_val = reg_val

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
        trace_df = self.get_model_data_for_trace(trace)
        title = "Trace {0} vs {1}-SVM Regression Predictions".format(
            trace.get_trace_id(), self._reg_val)
        self._plot_trace_data_vs_predictions(trace_df, title)

    def _initialize(self):
        """Initializes the support vector machine model.

        Returns
        -------
        None

        """
        super()._initialize()
        self._model = SVR(C=self._reg_val)
