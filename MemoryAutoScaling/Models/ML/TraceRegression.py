"""The `TraceRegression` class builds a linear regression model to
predict future time points for a time series.

"""
from MemoryAutoScaling.Models.ML import MLModel
from sklearn.linear_model import Ridge


class TraceRegression(MLModel):
    """A Linear Regression model for trace data.

    The regression model uses L2-regularization to ensure no coefficients
    are too large. This is used over L1-regularization becuase L1 tends to
    drive alot of coefficients to zero, but the set of parameters is quite
    small, so L2-regularization is used.

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
        super().__init__("TraceRegression", data_handler, lags)
        self._reg_val = reg_val

    def get_model_title(self):
        """A title describing the model.

        Returns
        -------
        str
            A string representing the title for the model.

        """
        return "{0}-{1}".format(self._reg_val, self._model_name)

    def _initialize(self):
        """Initializes the linear regression model.

        Returns
        -------
        None

        """
        super()._initialize()
        self._model = Ridge(alpha=self._reg_val)
