"""The `SequentialModel` class is an abstract base class used for all trace
models that predict sequentially. That is, these models use previous values
to predict the current value but do not store any state.

"""
from abc import abstractmethod
from MemoryAutoScaling import plotting, specs, utils
from MemoryAutoScaling.Models import TraceModel
from MemoryAutoScaling.Analysis import ModelResults


class SequentialModel(TraceModel):
    """Implements the functionality of a sequential trace model.

    Parameters
    ----------
    initial_pred: float
        A float representing the initial prediction for the model. This
        initial prediction is used until there is enough data available for
        the trace to make a prediction from trace data.
    train_prop: float
        A float in the range [0, 1] representing the proportion of
        observations in the training set. The default value is 0.6.
    val_prop: float
        A float in the range [0, 1] representing the proportion of data in
        the validation set. The default value is 0.2.
    max_mem: bool
        A boolean indicating if the target value is maximum memory. The
        default value is True, indicating that maximum memory usage is
        the target variable. Otherwise, maximum CPU usage is used.

    Attributes
    ----------
    _initial_pred: float
        The initial prediction for the model.
    _train_prop: float
        Represents the percent of data in the training set.
    _val_prop: float
        The proportion of data in the validation set.
    _max_mem: bool
        A boolean indicating the target variable. True indicates that maximum
        memory usage is the target variable. Otherwise, maximum CPU usage is
        used as the target.

    """
    def __init__(self, initial_pred, train_prop=0.6,
                 val_prop=0.2, max_mem=True):
        self._initial_pred = initial_pred
        self._train_prop = train_prop
        self._val_prop = val_prop
        self._max_mem = max_mem

    def get_target_variable(self):
        """The target variable for the model.

        Returns
        -------
        str
            A string indicating the target variable for the model.

        """
        return specs.get_target_variable(self._max_mem)

    def split_data(self, model_data, tuning=True):
        """Splits `model_data` into the training and testing set.

        If `tuning` is True, `model_data` is split into training and
        validation sets. Otherwise, it is split into training + validation
        and testing sets.

        Parameters
        ----------
        model_data: pandas.Object
            A pandas Object (Series or DataFrame) containing the data used to
            build the trace model.
        tuning: bool
            A boolean value indicating whether the split is for tuning.

        Returns
        -------
        pandas.Object, pandas.Object
            The two pandas objects obtained from `model_data` after applying
            the train-test split.

        """
        train_thresh, test_thresh = utils.calculate_split_thresholds(
            data, self._train_prop, self._val_prop, tuning)
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
        """Gets the data for modeling from `trace`.

        Parameters
        ----------
        trace: Trace
            The `Trace` being modeled.

        Returns
        -------
        np.array
            A numpy array representing the time series of maximum memory
            usage of the trace.

        """
        if self._max_mem:
            return trace.get_maximum_memory_time_series()
        return trace.get_maximum_cpu_time_series()

    def get_predictions_for_trace(self, trace, tuning=True):
        """Gets predictions for the training and testing set for `trace`.

        Parameters
        ----------
        trace: Trace
            The `Trace` for which predictions are retrieved.
        tuning: bool
            A boolean value indicating whether the predictions are for the
            validation set or the test set.

        Returns
        -------
        np.array, np.array
            Two numpy arrays representing the predictions for the training and
            testing sets on `trace`.

        """
        trace_ts = self.get_model_data_for_trace(trace)
        return self._get_predictions(trace_ts, tuning)

    def run_model_pipeline_for_trace(self, trace, tuning=True):
        """Runs the full modeling pipeline on `trace`.

        The modeling pipeline first obtains the data needed for modeling
        from `trace`. Next it splits this data into the training and testing
        sets. It then fits the model and obtains predictions. Lastly, these
        predictions are evaluated using MASE on the training and testing sets.
        The number of under predictions and the magnitude of the maximum under
        prediction for the test set are also calculated.

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
        y_train, y_test = self.split_data(trace_ts, tuning)
        total_spare = self.get_total_spare(trace, tuning)
        preds_train, preds_test = self._get_predictions(trace_ts)
        return ModelResults.from_data(
            self.get_params(), y_train, preds_train,
            y_test, preds_test, total_spare)

    def plot_trace_vs_prediction(self, trace, tuning=True):
        """Creates a plot of `trace` vs its predictions.

        The plot is arranged into two subplots. The first contains the maximum
        memory usage for the trace versus its prediction for the training set.
        The second plot is the same but for the testing set.

        Parameters
        ----------
        trace: Trace
            The `Trace` being plotted.
        tuning: bool
            A boolean value indicating whether the predictions are for the
            validation set or the test set.

        Returns
        -------
        None

        """
        trace_ts = self.get_model_data_for_trace(trace)
        title = "Trace {0} vs {1} Predictions".format(
            trace.get_trace_id(), self.get_model_title())
        self._plot_time_series_vs_prediction(trace_ts, title, tuning)

    @abstractmethod
    def _get_predictions(self, trace_ts, tuning=True):
        """Gets predictions for the training and testing set for `trace_ts`.

        Parameters
        ----------
        trace_ts: np.array
            A numpy array representing the time series for which predictions
            are calculated.
        tuning: bool
            A boolean value indicating whether the predictions are for the
            validation set or the test set.

        Returns
        -------
        np.array, np.array
            Two numpy arrays representing the predictions for the training and
            testing sets on `trace`.

        """
        pass

    def _plot_time_series_vs_prediction(self, time_series, title, tuning=True):
        """Plots `time_series` vs its prediction according to the model.

        The plot of `time_series` vs its predictions is divided into two
        subplots: one for the training set and another for the testing set.

        Parameters
        ----------
        time_series: np.array
            A numpy array representing the time series being plotted.
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
        y_train, y_test = self.split_data(time_series, tuning)
        preds_train, preds_test = self._get_predictions(time_series, tuning)
        plotting.plot_train_and_test_predictions_on_axes(
            y_train, preds_train, y_test, preds_test, (ax1, ax2), title)
