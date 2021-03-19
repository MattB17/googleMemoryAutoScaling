"""The `TimeSeriesRegression` class builds a linear regression model to
predict future time points for a time series.

"""
from MemoryAutoScaling.Models import MLModel
from sklearn.linear_model import LinearRegression


class TimeSeriesRegression(MLModel):
    """A LinearRegression model for time series data.

    Parameters
    ----------
    data_handler: MLDataHandler
        An `MLDataHandler` used to pre-process data for the model.

    Attributes
    ----------
    _model_name: str
        The name of the machine learning model.
    _data_handler: MLDataHandler
        The handler used to pre-process data for the model.
    _model: Object
        The underlying machine learning model being fit.
    _is_fit: bool
        Indicates if the model has been fitted to training data.

    """
    def __init__(self, data_handler):
        super().__init__("TimeSeriesRegression", data_handler)

    def initialize(self, **kwargs):
        """Initializes the linear regression model.

        Parameters
        ----------
        kwargs: dict
            Arbitrary keyword arguments used in initialization.

        Returns
        -------
        None

        """
        super().initialize()
        self._model = LinearRegression(**kwargs)
