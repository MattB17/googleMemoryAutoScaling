"""The `TraceARIMA` class is used to construct a predictive model based on
ARIMA. ARIMA models have 3 terms: `p`, `d`, and `q`. `p` indicates the auto
regressive component, that is the number of previous values used to predict
the current value. `d` refers to the number of differencing rounds that must
be applied to make the time series stationary. `q` is the moving average
component, refering to the number of lagged forecast errors used in the model.

"""
from MemoryAutoScaling import utils
from MemoryAutoScaling.Analysis import ModelResults
from MemoryAutoScaling.Models.Statistical import StatisticalModel
from statsmodels.tsa.statespace.sarimax import SARIMAX


class TraceARIMA(StatisticalModel):
    """An ARIMA model for time series data.

    Parameters
    ----------
    p: int
        An integer representing the autoregressive component of the model.
    d: int
        An integer representing the integrated component of the model.
        That is, the level of differencing needed to make the time
        series stationary.
    q: int
        An integer representing the moving average component of the model.
    train_prop: float
        A float in the range [0, 1] representing the proportion of data
        in the training set. The default value is 0.7

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
    def __init__(self, p, d, q, train_prop=0.7):
        super().__init__(train_prop)
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

    def get_model_title(self):
        """The title for the model.

        Returns
        -------
        str
            A string representing the title for the model.

        """
        return "Trace ARIMA {}".format(self.get_params())

    def split_data(self, model_data):
        """Splits `model_data` into training and testing time series.

        Parameters
        ----------
        model_data: np.array
            A numpy array containing the data of the time series being
            forecast.

        Returns
        -------
        np.array, np.array
            Two numpy arrays representing the records in the train and
            test sets respectively.

        """
        train_cutoff = utils.get_train_cutoff(model_data, self._train_prop)
        return model_data[:train_cutoff], model_data[train_cutoff:]

    def get_model_data_for_trace(self, trace):
        """Retrieves the data for modeling from `trace`.

        Parameters
        ----------
        trace: Trace
            The `Trace` object being modeled.

        Returns
        -------
        np.array
            A numpy array representing the maximum memory usage for `trace`
            for each time interval.

        """
        return trace.get_maximum_memory_time_series()

    def run_model_pipeline_for_trace(self, trace):
        """Runs the full modeling pipeline on `trace`.

        The modeling pipeline first obtains the data needed for modeling
        from `trace`. Next it splits this data into the training and testing
        sets. It then fits the model and obtains predictions. Lastly, these
        predictions are evaluated using MASE on the training and testing sets.

        Parameters
        ----------
        trace: Trace
            The `Trace` being modeled.

        Returns
        -------
        ModelResults
            A `ModelResults` object containing the results of building the
            model on `trace`.

        """
        trace_ts = self.get_model_data_for_trace(trace)
        train_ts, test_ts = self.split_data(trace_ts)
        self._fit(train_ts)
        preds_train, preds_test = self._get_predictions(len(test_ts))
        return ModelResults(
            self.get_params(), train_ts, preds_train, test_ts, preds_test)

    def plot_trace_vs_prediction(self, trace):
        """Creates a plot of `trace` vs its prediction.

        The plot is divided into two subplots showing the actual values versus
        the predicted values for the training and testing sets, respectively.

        Parameters
        ----------
        trace: Trace
            The `Trace` object for which actual and predicted values are
            plotted.

        Returns
        -------
        None

        """
        fig, (ax1, ax2) = plt.subplots(2)
        trace_ts = self.get_model_data_for_trace(trace)
        train_ts, test_ts = self.split_data(trace_ts)
        preds_train, preds_test = self._get_predictions(len(test_ts))
        utils.plot_train_and_test_predictions_on_axes(
            train_ts, preds_train, test_ts, preds_test,
            (ax1, ax2), self.get_plot_title())


    def _fit(self, train_data):
        """Fits the ARIMA model based on `train_data`.

        Parameters
        ----------
        train_data: np.array
            A numpy array representing the time series used for training.

        Returns
        -------
        None

        """
        model = SARIMAX(
            train_data, order=self.get_params(), simple_differencing=False)
        self._model = model.fit(disp=False)
