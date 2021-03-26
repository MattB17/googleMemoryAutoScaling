"""The `TimeSeriesXGB` class builds an additive regression model using
`xgboost`. This additive regression model recursively builds small regression
models, in which the current regression model puts more weight on observations
for which the previous models are bad at predicting. These simple regression
models are then added together.

"""
from MemoryAutoScaling.Models import MLModel
from xgboost import XGBRegressor


class TimeSeriesXGB(MLModel):
    """An XGBoost model for time series data.

    Parameters
    ----------
    data_handler: MLDataHandler
        An `MLDataHandler` used to preprocess data for the model.
    lag: int
        An integer representing the lag to use for time series features to
        the model. That is, for any time series used as a feature to the
        model, the values of that time series lagged by `lag` will be the
        feature used in the regression.
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
    _lag: int
        The lag used for time series features to the model.
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
    def __init__(data_handler, lag, learning_rate, estimators, depth):
        super().__init__("TimeSeriesXGB", data_handler, lag)
        self._learning_rate = learning_rate
        self._estimators = estimators
        self._depth = depth

    def initialize(self, **kwargs):
        """Initializes the XGBoost Regression model.

        Parameters
        ----------
        kwargs: dict
            Arbitrary keyword arguments used in initialization.

        Returns
        -------
        None

        """
        super().initialize()
        self._model = XGBRegressor(learning_rate=self._learning_rate,
                                   max_depth=self._depth,
                                   n_estimators=self._estimators,
                                   **kwargs)

    def get_train_and_test_predictions(self, train_features, test_features):
        """Retrieves predictions for the training and testing sets.

        Parameters
        ----------
        train_features: pd.DataFrame
            A pandas DataFrame representing the values for the features of the
            training set.
        test_features: pd.DataFrame
            A pandas DataFrame representing the values for the features of the
            testing set.

        Returns
        -------
        np.array, np.array
            Two numpy arrays representing the predictions for the training and
            testing sets, respectively.

        """
        train_preds = self.get_predictions(train_features)
        test_preds = self.get_predictions(test_features)
        return train_preds, test_preds
