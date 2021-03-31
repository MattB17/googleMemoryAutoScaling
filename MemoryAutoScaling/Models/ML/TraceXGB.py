"""The `TraceXGB` class builds an additive regression model using `xgboost`.
This additive regression model recursively builds small regression models, in
which the current regression model puts more weight on observations for which
the previous models are bad at predicting. These simple regression models are
then added together.

"""
from MemoryAutoScaling.Models.ML import MLModel
from xgboost import XGBRegressor


class TraceXGB(MLModel):
    """An XGBoost model for time series data.

    Parameters
    ----------
    data_handler: MLDataHandler
        An `MLDataHandler` used to preprocess data for the model.
    lags: list
        A list of integers representing the lags to use for time series
        features to the model. That is, for any time series used as a feature
        to the model, the model will use the value of the time series lagged
        by the time points in `lags` when predicting for the target variable.
    learning_rate: float
        A float in the range [0, 1], denoting the learning rate.
    estimators: int
        An integer denoting the number of additive components of the model.
    depth: int
        An integer denoting the depth of each model. Each model is a simple
        tree structure of height `depth`.

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
    _learning_rate: float
        The learning rate for the model.
    _estimators: int
        The number of additive components of the model.
    _depth: int
        The depth of the model.

    """
    def __init__(self, data_handler, lags, learning_rate, estimators, depth):
        super().__init__("TraceXGB", data_handler, lags)
        self._learning_rate = learning_rate
        self._estimators = estimators
        self._depth = depth

    def get_params(self):
        """Returns the parameters of the model.

        The parameters of the model consists of a three element tuple which
        gives the learning rate, number of estimators, and max depth.

        Returns
        -------
        tuple
            A 3-element tuple containing the model parameters.

        """
        return self._learning_rate, self._estimators, self._depth

    def get_model_title(self):
        """A title describing the model.

        Returns
        -------
        str
            A string representing the title for the model.

        """
        return "{0}-{1}".format(self.get_params(), self._model_name)

    def _initialize(self):
        """Initializes the XGBoost Regression model.

        Returns
        -------
        None

        """
        super()._initialize()
        self._model = XGBRegressor(learning_rate=self._learning_rate,
                                   max_depth=self._depth,
                                   n_estimators=self._estimators,
                                   objective='reg:squarederror')
