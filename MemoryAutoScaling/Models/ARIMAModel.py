"""The `ARIMAModel` class is used to construct a predictive model based on
ARIMA. ARIMA models have 3 terms: `p`, `d`, and `q`. `p` indicates the auto
regressive component, that is the number of previous values used to predict
the current value. `d` refers to the number of differencing rounds that must
be applied to make the time series stationary. `q` is the moving average
component, refering to the number of lagged forecast errors used in the model.

"""
from sklearn.metrics import mean_squared_error
from MemoryAutoScaling import utils
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ARIMAModel:
    """An ARIMA model for time series data.

    Parameters
    ----------
    train_prop: float
        A float in the range [0, 1] representing the proportion of data
        in the training set.
    p: int
        An integer representing the autoregressive component of the model.
    d: int
        An integer representing the integrated component of the model.
        That is, the level of differencing needed to make the time
        series stationary.
    q: int
        An integer representing the moving average component of the model.

    Attributes
    ----------
    _train_prop: float
        The proportion of data in the training set.
    _model: Object
        The underlying machine learning model being fit.
    _p: int
        The autoregressive component of the model.
    _d: int
        The integrated component of the model.
    _q: int
        The moving average component of the model.

    """
    def __init__(self, train_prop, p, d, q):
        self._train_prop = train_prop
        self._model = None
        self._p = p
        self._d = d
        self._q = q

    def get_params(self):
        """Returns the order for the ARIMA model.

        The order is the three element tuple `(p, d, q)` representing
        the autoregressive component, the degree of differencing, and the
        moving average component, respectively.

        Returns
        -------
        tuple
            A three element tuple of integers representing the order of the
            ARIMA model.

        """
        return self._p, self._d, self._q

    def split_data(self, data_trace):
        """Splits `data_trace` into training and testing time series.

        Parameters
        ----------
        data_trace: np.array
            A numpy array containing the data of the time series being
            forecast.

        Returns
        -------
        np.array, np.array
            Two numpy arrays representing the records in the train and
            test sets respectively.

        """
        train_cutoff = utils.get_train_cutoff(data_trace, self._train_prop)
        return data_trace[:train_cutoff], data_trace[train_cutoff:]

    def fit(self, train_trace):
        """Fits the model based on training time series, `train_trace`.

        Parameters
        ----------
        train_trace: np.array
            A numpy array representing the time series used for training.

        Returns
        -------
        None

        """
        model = SARIMAX(
            train_trace, order=self.get_order(), simple_differencing=False)
        self._model = model.fit(disp=False)

    def get_predictions(self, test_trace):
        """Retrieves model predictions for `test_trace`.

        Parameters
        ----------
        test_trace: np.array
            A numpy array representing the time series used for testing.

        Returns
        -------
        SARIMAX.prediction
            A `SARIMAX.prediction` object containing the predictions
            for the time series up to the end of the testing period.

        """
        forecast_len = len(test_trace)
        return self._model.get_prediction(
            end=self._model.nobs + forecast_len - 1)

    def run_model_pipeline(self, train_trace, test_trace):
        """Runs the model pipeline on `train_trace` and `test_trace`.

        The model is instantiated and fit based on `train_trace`. Then
        predictions are obtained for the test time frame and these
        predictions are compared to `test_trace` using the MSE.

        Parameters
        ----------
        train_trace: np.array
            A numpy array representing the proportion of the time series
            corresponding to the training set.
        test_trace: np.array
            A numpy array representing the proportion of the time series
            corresponding to the testing set.

        Returns
        -------
        SARIMAX.prediction, float, float
            A `SARIMAX.prediction` object containing the predictions for
            the time series up to the end of the testing period and two
            floats corresponding to the MSE of the predictions for the
            training and testing sets respectively.

        """
        self.fit(train_trace)
        preds = self.get_predictions(test_trace)
        n_forecast = len(test_trace)
        mean_preds = utils.impute_for_time_series(preds.predicted_mean, 0)
        train_mse = mean_squared_error(
            train_trace, mean_preds[:-n_forecast])
        test_mse = mean_squared_error(
            test_trace, mean_preds[-n_forecast:])
        return preds, train_mse, test_mse

    def run_model_pipeline_for_trace(self, data_trace):
        """Runs the model pipeline on `data_trace`.

        `data_trace` is first split into a training and testing set and then
        the model pipeline is run on these sets..

        Parameters
        ----------
        data_trace: Trace
            A `Trace` object representing the time series being modelled.

        Returns
        -------
        SARIMAX.prediction, float, float
            A `SARIMAX.prediction` object containing the predictions for
            the time series up to the end of the testing period and two
            floats corresponding to the MSE of the predictions for the
            training and testing sets respectively.

        """
        trace_ts = data_trace.get_maximum_memory_time_series()
        train_trace, test_trace = self.split_data(trace_ts)
        return self.run_model_pipeline(train_trace, test_trace)
