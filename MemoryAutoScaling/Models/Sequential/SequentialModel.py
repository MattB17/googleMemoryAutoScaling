"""The `SequentialModel` class is an abstract base class used for all trace
models that predict sequentially. That is, these models use previous values
to predict the current value but do not store any state.

"""
from abc import abstractmethod
from MemoryAutoScaling import utils
from MemoryAutoScaling.Models import TraceModel


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
        observations in the training set. The default value is 0.7.

    Attributes
    ----------
    _initial_pred: float
        The initial prediction for the model.
    _train_prop: float
        Represents the percent of data in the training set.

    """
    def __init__(self, initial_pred, train_prop=0.7):
        self._initial_pred = initial_pred
        self._train_prop = train_prop

    def split_data(self, model_data):
        """Splits `model_data` into the training and testing set.

        Parameters
        ----------
        model_data: pandas.Object
            A pandas Object (Series or DataFrame) containing the data used to
            build the trace model.

        Returns
        -------
        pandas.Object, pandas.Object
            The two pandas objects obtained from `model_data` after applying
            the train-test split.

        """
        train_cutoff = utils.get_train_cutoff(model_data, self._train_prop)
        return model_data[:train_cutoff], model_data[train_cutoff:]

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
        return trace.get_maximum_memory_time_series()

    def get_predictions_for_trace(self, trace):
        """Gets predictions for the training and testing set for `trace`.

        Parameters
        ----------
        trace: Trace
            The `Trace` for which predictions are retrieved.

        Returns
        -------
        np.array, np.array
            Two numpy arrays representing the predictions for the training and
            testing sets on `trace`.

        """
        trace_ts = self.get_model_data_for_trace(trace)
        return self._get_predictions(trace_ts)

    def run_model_pipeline_for_trace(self, trace):
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

        Returns
        -------
        tuple
            A tuple of six floats. The first two represent the mean absolute
            scaled error for the training and testing sets, respectively.
            The next two represent the proportion of under predictions and the
            magnitude of the maximum under prediction, respectively. The last
            two represent the proportion of over predictions and the magnitude
            of the average over prediction.

        """
        trace_ts = self.get_model_data_for_trace(trace)
        y_train, y_test = self.split_data(trace_ts)
        preds_train, preds_test = self._get_predictions(trace_ts)
        return utils.calculate_evaluation_metrics(
            y_train, preds_train, y_test, preds_test)

    def plot_trace_vs_prediction(self, trace):
        """Creates a plot of `trace` vs its predictions.

        The plot is arranged into two subplots. The first contains the maximum
        memory usage for the trace versus its prediction for the training set.
        The second plot is the same but for the testing set.

        Parameters
        ----------
        trace: Trace
            The `Trace` being plotted.

        Returns
        -------
        None

        """
        trace_ts = self.get_model_data_for_trace(trace)
        title = "Trace {0} vs {1} Predictions".format(
            trace.get_trace_id(), self.get_model_title())
        self._plot_time_series_vs_prediction(trace_ts, title)

    @abstractmethod
    def _get_predictions(self, trace_ts):
        """Gets predictions for the training and testing set for `trace_ts`.

        Parameters
        ----------
        trace_ts: np.array
            A numpy array representing the time series for which predictions
            are calculated.

        Returns
        -------
        np.array, np.array
            Two numpy arrays representing the predictions for the training and
            testing sets on `trace`.

        """
        pass

    def _plot_time_series_vs_prediction(self, time_series, title):
        """Plots `time_series` vs its prediction according to the model.

        The plot of `time_series` vs its predictions is divided into two
        subplots: one for the training set and another for the testing set.

        Parameters
        ----------
        time_series: np.array
            A numpy array representing the time series being plotted.
        title: str
            A string representing the title of the plot.

        Returns
        -------
        None

        """
        fig, (ax1, ax2) = plt.subplots(2)
        y_train, y_test = self.split_data(time_series)
        preds_train, preds_test = self._get_predictions(time_series)
        utils.plot_train_and_test_predictions_on_axes(
            y_train, preds_train, y_test, preds_test, (ax1, ax2), title)
