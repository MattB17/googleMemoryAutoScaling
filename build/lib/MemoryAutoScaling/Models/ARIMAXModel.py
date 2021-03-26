"""The `ARIMAXModel` class is used to construct a predictive model based on
the ARIMAX model. ARIMAX models are the combination of an ARIMA model and
a regression model on explanatory features.

"""
from statsmodels.tsa.statespace.sarimax import SARIMAX
from MemoryAutoScaling.Models import MLModel


class ARIMAXModel(MLModel):
    """An `ARIMAX` model for data traces.

    Parameters
    ----------
    data_handler: MLDataHandler
        An `MLDataHandler` used to pre-process data for the model.
    lag: int
        An integer representing the lag to use for time series features to
        the model. That is, for any time series used as a feature to the
        model, the values of that time series lagged by `lag` will be a
        feature used in the regression component of the ARIMAX model.
    p: int
        An integer representing the autoregressive component of the model.
    d: int
        An integer representing the integrated component of the model. That
        is, the level of differencing needed to make the time series
        stationary.
    q: int
        An integer representing the moving average component of the model.

    Attributes
    ----------
    _model_name: str
        The name of the machine learning model.
    _data_handler: MLDataHandler
        The handler used to pre-process data for the model.
    _lag: int
        The lag used for time series features to the model.
    _p: int
        The autoregressive component of the model.
    _d: int
        The integrated component of the model.
    _q: int
        The moving average component of the model.
    _model: SARIMAX
        The underlying SARIMAX model being fit to the data. A value of None
        indicates that no model is currently fit to the data.
    _is_fit: bool
        Indicates if the model has been fit to the training data.

    """
    def __init__(self, data_handler, lag, p, d, q):
        super().__init__("ARIMAXModel", data_handler, lag)
        self._p = p
        self._d = d
        self._q = q

    def get_order(self):
        """Returns the order for the ARIMA component of the model.

        The order is the three element tuple `(p, d, q)` representing the
        autoregressive component, the degree of differencing, and the moving
        average component, respectively.

        Returns
        -------
        tuple
            A three element tuple of integers representing the order of the
            ARIMA component of the model.

        """
        return self._p, self._d, self._q

    def fit(self, train_features, train_target):
        """Fits the model based on `train_features` and `train_target`.

        An ARIMAX model is built to predict the target variable with data
        given by `train_target` based on the features with data given by
        `train_features`.

        Parameters
        ----------
        train_features: pd.DataFrame
            A pandas DataFrame representing the training features.
        train_target: pd.Series
            A pandas Series representing the target variable.

        Returns
        -------
        None

        """
        model = SARIMAX(
            train_target, train_features,
            order=self.get_order(), simple_differencing=False)
        self._model = model.fit(disp=False)
        self._is_fit = True

    def get_predictions(self, test_features):
        """Retrieves model predictions for `test_features`.

        Parameters
        ----------
        test_features: pd.DataFrame
            A pandas DataFrame rerpresenting the testing features.

        Returns
        -------
        np.array
            A numpy array containing the predictions up to the end of the
            testing period.

        """
        if self._is_fit:
            forecast_len = len(test_features) - 1
            preds = self._model.get_prediction(
                end=self._model.nobs + forecast_len, exog=test_features)
            return preds.predicted_mean

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
            testing sets respectively.


        """
        preds = self.get_predictions(test_features)
        train_cutoff = len(train_features)
        return preds[:train_cutoff], preds[train_cutoff:]
