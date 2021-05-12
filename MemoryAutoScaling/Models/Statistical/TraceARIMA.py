"""The `TraceARIMA` class is used to construct a predictive model based on
ARIMA. ARIMA models have 3 terms: `p`, `d`, and `q`. `p` indicates the auto
regressive component, that is the number of previous values used to predict
the current value. `d` refers to the number of differencing rounds that must
be applied to make the time series stationary. `q` is the moving average
component, refering to the number of lagged forecast errors used in the model.

"""
from MemoryAutoScaling import plotting, specs, utils
from MemoryAutoScaling.Evaluation import ModelResults
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
        in the training set. The default value is 0.6.
    val_prop: float
        A float in the range [0, 1] representing the proportion of data in
        the validation set. The default value is 0.2.
    max_mem: bool
        A boolean indicating if the target value is maximum memory. The
        default value is True, indicating that maximum memory usage is
        the target variable. Otherwise, maximum CPU usage is used.

    Attributes
    ----------
    _train_prop: float
        The proportion of data in the training set.
    _val_prop: float
        The proportion of data in the validation set.
    _model: Object
        The underlying machine learning model being fit.
    _p: int
        The autoregressive component of the model.
    _d: int
        The integrated component of the model.
    _q: int
        The moving average component of the model.
    _max_mem: bool
        A boolean indicating the target variable. True indicates that maximum
        memory usage is the target variable. Otherwise, maximum CPU usage is
        used as the target.

    """
    def __init__(self, p, d, q, train_prop=0.6, val_prop=0.2, max_mem=True):
        super().__init__(train_prop, val_prop)
        self._p = p
        self._d = d
        self._q = q
        self._max_mem = max_mem

    def get_target_variable(self):
        """The target variable for the model.

        Returns
        -------
        str
            A string indicating the target variable for the model.

        """
        return specs.get_target_variable(self._max_mem)

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
        return {'p': self._p, 'd': self._d, 'q': self._q}

    def get_model_title(self):
        """The title for the model.

        Returns
        -------
        str
            A string representing the title for the model.

        """
        return "Trace ARIMA {}".format(self.get_params())

    def split_data(self, model_data, tuning=True):
        """Splits `model_data` into training and testing time series.

        If `tuning` is True, `model_data` is split into training and
        validation sets. Otherwise, it is split into training + validation
        and testing sets.

        Parameters
        ----------
        model_data: np.array
            A numpy array containing the data of the time series being
            forecast.
        tuning: bool
            A boolean value indicating whether the split is for tuning.

        Returns
        -------
        np.array, np.array
            Two numpy arrays representing the records in the train and
            test sets respectively.

        """
        train_thresh, test_thresh = utils.calculate_split_thresholds(
            model_data, self._train_prop, self._val_prop, tuning)
        return model_data[:train_thresh], model_data[train_thresh:test_thresh]

    def get_total_spare(self, trace, tuning=True):
        """The spare amount of the target for `trace` over the test window.

        Parameters
        ----------
        trace: Trace
            The `Trace` for which the spare is calculated.
        tuning: bool
            A boolean value indicating whether the spare is being calculated
            to tune the model or evaluate the model on the test set.

        Returns
        -------
        float
            A float representing the total spare amount of the target
            variable for `trace` over the test window.

        """
        target_ts = trace.get_target_time_series(self.get_target_variable())
        train_thresh, test_thresh = utils.calculate_split_thresholds(
            target_ts, self._train_prop, self._val_prop, tuning)
        return trace.get_spare_resource_in_window(
            self.get_target_variable(), train_thresh, test_thresh)

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
        if self._max_mem:
            return trace.get_maximum_memory_time_series()
        return trace.get_maximum_cpu_time_series()

    def run_model_pipeline_for_trace(self, trace, tuning=True):
        """Runs the full modeling pipeline on `trace`.

        The modeling pipeline first obtains the data needed for modeling
        from `trace`. Next it splits this data into the training and testing
        sets. It then fits the model and obtains predictions. Lastly, these
        predictions are evaluated using MASE on the training and testing sets.

        Parameters
        ----------
        trace: Trace
            The `Trace` being modeled.
        tuning: bool
            A boolean value indicating whether the model is being tuned on
            the validation set or evaluated on the test set.

        Returns
        -------
        ModelResults
            A `ModelResults` object containing the results of building the
            model on `trace`.

        """
        trace_ts = self.get_model_data_for_trace(trace)
        train_ts, eval_ts = self.split_data(trace_ts, tuning)
        total_spare = self.get_total_spare(trace, tuning)
        self._fit(train_ts)
        preds_train, preds_eval = self._get_predictions(len(eval_ts))
        return ModelResults.from_data(
            self.get_params(), train_ts, preds_train, eval_ts,
            preds_eval, trace, self.get_target_variable())

    def plot_trace_vs_prediction(self, trace, tuning=True):
        """Creates a plot of `trace` vs its prediction.

        The plot is divided into two subplots showing the actual values versus
        the predicted values for the training and testing sets, respectively.

        Parameters
        ----------
        trace: Trace
            The `Trace` object for which actual and predicted values are
            plotted.
        tuning: bool
            A boolean value indicating whether the predictions are for the
            validation set or the test set.

        Returns
        -------
        None

        """
        fig, (ax1, ax2) = plt.subplots(2)
        trace_ts = self.get_model_data_for_trace(trace)
        train_ts, eval_ts = self.split_data(trace_ts, tuning)
        preds_train, preds_eval = self._get_predictions(len(eval_ts))
        plotting.plot_train_and_test_predictions_on_axes(
            train_ts, preds_train, eval_ts, preds_eval,
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
        arima_order = (self._p, self._d, self._q)
        model = SARIMAX(
            train_data, order=arima_order, simple_differencing=False)
        self._model = model.fit(disp=False)
