"""The `TraceARIMAX` class is used to construct a predictive model based on
the ARIMAX model. ARIMAX models are the combination of an ARIMA model and
a regression model on explanatory features.

"""
from MemoryAutoScaling import plotting
from MemoryAutoScaling.Models.ML import MLBase
from MemoryAutoScaling.Analysis import ModelResults
from statsmodels.tsa.statespace.sarimax import SARIMAX


class TraceARIMAX(MLBase):
    """An `ARIMAX` model for data traces.

    Parameters
    ----------
    data_handler: MLDataHandler
        An `MLDataHandler` used to pre-process data for the model.
    lags: list
        A list of integers representing the lags to use for time series
        features to the model. That is, for any time series used as a feature
        to the model, the model will use the value of the time series lagged
        by the time points in `lags` when predicting for the target variable.
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
    _lags: list
        The lags used for time series features to the model.
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
    def __init__(self, data_handler, lags, p, d, q):
        super().__init__("TraceARIMAX", data_handler, lags)
        self._p = p
        self._d = d
        self._q = q

    def get_target_variable(self):
        """The target variable for the model.

        Returns
        -------
        str
            A string indicating the target variable for the model.

        """
        return self._data_handler.get_target_variables()[0]

    def get_params(self):
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
        return {'p': self._p, 'd': self._d, 'q': self._q}

    def get_model_title(self):
        """A title describing the model.

        Returns
        -------
        str
            A string representing the title for the model.

        """
        return "{0}-{1}".format(self.get_params(), self._model_name)

    def _fit(self, train_features, train_target):
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
        arimax_order = (self._p, self._d, self._q)
        model = SARIMAX(
            train_target, train_features,
            order=arimax_order, simple_differencing=False)
        self._model = model.fit(disp=False)
        self._is_fit = True

    def fit_and_get_test_predictions(self, trace, tuning=True):
        """Fits the model and gets test predictions for `trace`.

        Parameters
        ----------
        trace: Trace
            The `Trace` for which predictions are retrieved.
        tuning: bool
            A boolean value indicating whether the model is being tuned on
            the validation set or evaluated on the test set.

        Returns
        -------
        np.array, np.array
            A numpy array representing the actual values and predictions for
            the testing set of `trace`.

        """
        trace_df = self.get_model_data_for_trace(trace)
        X_train, y_train, X_test, y_test = self.split_data(trace_df, tuning)
        self._fit(X_train, y_train)
        _, test_preds = self._get_train_and_test_predictions(X_train, X_test)
        return y_test, test_preds

    def _get_predictions(self, test_features):
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

    def _get_train_and_test_predictions(self, train_features, test_features):
        """Gets predictions from `train_features` and `test_features`.

        Parameters
        ----------
        train_features: pd.DataFrame
            A pandas DataFrame containing the features for the training set.
        test_features: pd.DataFrame
            A pandas DataFrame containing the features for the testing set.

        Returns
        -------
        np.array, np.array
            Two numpy arrays representing the predictions for the training and
            testing sets on `trace`.

        """
        preds = self._get_predictions(test_features)
        train_cutoff = len(train_features)
        return preds[:train_cutoff], preds[train_cutoff:]

    def _run_model_pipeline(self, X_train, train_target,
                            X_test, test_target, trace):
        """Runs the model pipeline on the training and testing data.

        The model is instantiated and then fit on `X_train` and
        `train_target`. Predictions are made on `X_test` and these
        predictions are compared to `test_target` using the mean squared
        error.

        Parameters
        ----------
        X_train: pd.DataFrame
            A pandas DataFrame representing the features for the training set.
        train_target: pd.Series
            A pandas Series representing the target variable for the
            training set.
        X_test: pd.DataFrame
            A pandas Dataframe representing the features for the testing set.
        test_target: pd.Series
            A pandas Series representing the target variable for the testing
            set.
        trace: Trace
            The `Trace` on which the model pipeline is being run.

        Returns
        -------
        ModelResults
            A `ModelResults` object containing the results of building the
            model on `trace`.

        """
        self._fit(X_train, train_target)
        train_preds, test_preds = self._get_train_and_test_predictions(
            X_train, X_test)
        return ModelResults.from_data(
            self.get_params(), train_target, train_preds, test_target,
            test_preds, trace, self.get_target_variable())

    def _plot_trace_data_vs_predictions(self, trace_df, title, tuning=True):
        """Plots the target time series of `trace_df` vs its model prediction.

        The plot of the time series vs its predictions is divided into two
        subplots: one for the training set and another for the testing set.

        Parameters
        ----------
        trace_df: pd.DataFrame
            A pandas DataFrame containing the trace data used for modeling.
        title: str
            A string representing the title of the plot.
        tuning: bool
            A boolean value indicating whether the predictions are for the
            validation set or the test set.

        Returns
        -------
        None

        """
        fig, (ax1, ax2) = plt.subplots(2)
        X_train, y_train, X_eval, y_eval = self.split_data(trace_df, tuning)
        preds_train, preds_eval = self._get_train_and_test_predictions(
            X_train, X_eval)
        plotting.plot_train_and_test_predictions_on_axes(
            y_train, preds_train, y_eval, preds_eval, (ax1, ax2), title)
