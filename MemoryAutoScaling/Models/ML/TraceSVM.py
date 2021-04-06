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
        super().__init__("TraceSVM", data_handler, lags)
        self._reg_val = reg_val

    def get_params(self):
        """The parameters of the model.

        The model parameters correspond to the regularization parameter.

        Returns
        -------
        tuple
            A 1 element tuple containing the model parameters, corresponding
            to the regularization parameter.

        """
        return (self._reg_val,)

    def get_model_title(self):
        """A title describing the model.

        Returns
        -------
        str
            A string representing the title for the model.

        """
        return "{0}-{1}".format(self._reg_val, self._model_name)

    def _initialize(self):
        """Initializes the support vector machine model.

        Returns
        -------
        None

        """
        super()._initialize()
        self._model = SVR(C=self._reg_val)
