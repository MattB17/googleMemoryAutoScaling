"""The `TraceAR` class is used to construct a predictive model based on
Auto Regression. Auto Regressive models have one term, `p` specifying the
number of previous values used to predict the current value.

"""
from MemoryAutoScaling import plotting, specs, utils
from MemoryAutoScaling.Analysis import ModelResults
from MemoryAutoScaling.Models.Statistical import StatisticalModel
from statsmodels.tsa.statespace.sarimax import SARIMAX


class TraceAR(StatisticalModel):
    """An auto regression model for time series data.

    Parameters
    ----------
    p: int
        An integer representing the autoregressive component of the model.
    train_prop: float
        A float in the range [0, 1] representing the proportion of data
        in the training set. The default value is 0.7
    max_mem: bool
        A boolean indicating if the target value is maximum memory. The
        default value is True, indicating that maximum memory usage is
        the target variable. Otherwise, maximum CPU usage is used.

    Attributes
    ----------
    _train_prop: float
        The proportion of data in the training set.
    _model: Object
        The underlying machine learning model being fit.
    _p: int
        The autoregressive component of the model.
    _max_mem: bool
        A boolean indicating the target variable. True indicates that maximum
        memory usage is the target variable. Otherwise, maximum CPU usage is
        used as the target.

    """
    def __init__(self, p, train_prop=0.7, max_mem=True):
        super().__init__(train_prop)
        self._p = p
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
        return (self._p, )

    def get_model_title(self):
        """The title for the model.

        Returns
        -------
        str
            A string representing the title for the model.

        """
        return "Trace Auto Regression {}".format(self.get_params())

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
        if self._max_mem:
            return trace.get_maximum_memory_time_series()
        return trace.get_maximum_cpu_time_series()

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
        return ModelResults.from_data(
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
        plotting.plot_train_and_test_predictions_on_axes(
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
        ar_order = (self._p, 0, 0)
        model = SARIMAX(
            train_data, order=ar_order, simple_differencing=False)
        self._model = model.fit(disp=False)