"""The `TraceVARMAX` class is used to construct a predictive model based on
the VARMAX model. VARMAX models are the combination of a VARMA model and
a regression model on explanatory features.

"""
from MemoryAutoScaling import parallel, plotting
from MemoryAutoScaling.Models.ML import MLBase
from MemoryAutoScaling.Analysis import ModelResults
from statsmodels.tsa.statespace.varmax import VARMAX


class TraceVARMAX(MLBase):
    """A `VARMAX` model for data traces.

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
    _q: int
        The moving average component of the model.
    _model: VARMAX
        The underlying VARMAX model being fit to the data. A value of None
        indicates that no model is currently fit to the data.
    _is_fit: bool
        Indicates if the model has been fit to the training data.

    """
    def __init__(self, data_handler, lags, p, q):
        super().__init__("TraceVARMAX", data_handler, lags)
        self._p = p
        self._q = q

    def get_params(self):
        """Returns the order for the VARMA component of the model.

        The order is the two element tuple `(p, q)` representing the
        autoregressive component and the moving average component,
        respectively.

        Returns
        -------
        tuple
            A two element tuple of integers representing the order of the
            ARIMA component of the model.

        """
        return {'p': self._p, 'q': self._q}

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

        A VARMAX model is built to predict the target variables with data
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
        varmax_order = (self._p, self._q)
        model = VARMAX(train_target, train_features, order=varmax_order)
        self._model = model.fit(disp=False)
        self._is_fit = True

    def _get_predictions(self, test_features):
        """Retrieves model predictions for `test_features`.

        Parameters
        ----------
        test_features: pd.DataFrame
            A pandas DataFrame rerpresenting the testing features.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the predictions up to the end of the
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
        pd.DataFrame, pd.DataFrame
            Two pandas DataFrames arrays representing the predictions for the
            training and testing sets on `trace`.

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
        return parallel.get_multivariate_model_results(
            self.get_params(), train_target, train_preds, test_target,
            test_preds, trace, self._data_handler.get_target_variables())

    def _plot_trace_data_vs_predictions(self, trace_df, title, tuning=True):
        """Plots the target time series of `trace_df` vs its model prediction.

        The plot of the time series vs its predictions for each model
        variable. Each variable is plotted in a row and the row is divided
        into two subplots: one for the training set and another for the
        testing set.

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
        var_count = len(self._data_handler.get_target_variables())
        fig, axes = plt.subplots(var_count, 2, figsize=(20, 10*var_count))
        trace_df = self.get_model_data_for_trace(trace)
        X_train, y_train, X_eval, y_eval = self.split_data(trace_df, tuning)
        train_preds, eval_preds = self._get_train_and_test_predictions(
            X_train, X_eval)
        plotting.plot_multivariate_train_and_test_predictions(
            y_train, train_preds, y_eval, eval_preds,
            axes, self._data_handler.get_target_variables(),
            self.get_plot_title())
