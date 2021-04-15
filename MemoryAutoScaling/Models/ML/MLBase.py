"""The `MLBase` class is an abstract base class used for all machine
learning models that predict using a time series. It represents an interface
providing the basic framework used by all machine learning models.

"""
import numpy as np
from abc import abstractmethod
from MemoryAutoScaling import utils
from MemoryAutoScaling.Models import TraceModel
from MemoryAutoScaling.Analysis import ModelResults
from MemoryAutoScaling.DataHandling import MLDataHandler



class MLBase(TraceModel):
    """Used to predict future values of a time series.

    The model has an `MLDataHandler`, `data_handler`, to ensure that all
    data handled by the model is handled consistently.

    Parameters
    ----------
    model_name: str
        A string representing the name of the machine learning model.
    data_handler: MLDataHandler
        An `MLDataHandler` used to pre-process data for the model.
    lags: list
        A list of integers representing the lags to use for time series
        features to the model. That is, for any time series used as a feature
        to the model, the model will use the value of the time series lagged
        by the time points in `lags` when predicting for the target variable.

    Attributes
    ----------
    _model_name: str
        The name of the machine learning model.
    _data_handler: MLDataHandler
        The handler used to pre-process data for the model.
    _lags: list
        The lags used for time series features to the model.
    _model: Object
        The underlying machine learning model being fit.
    _is_fit: bool
        Indicates if the model has been fitted to training data.

    """
    def __init__(self, model_name, data_handler, lags):
        self._model_name = model_name
        self._data_handler = data_handler
        self._lags = lags
        self._model = None
        self._is_fit = False

    @abstractmethod
    def get_model_title(self):
        """A title describing the model.

        Returns
        -------
        str
            A string representing the title for the model.

        """
        pass

    def split_data(self, data):
        """Splits data into features and targets for the train and test sets.

        Parameters
        ----------
        data: pd.DataFrame
            A pandas DataFrame containing the data on which the split is
            performed.

        Returns
        -------
        pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
            A pandas DataFrame and Series representing the training set values
            of the features and target variable, respectively. The second
            DataFrame and Series represent the same split for the testing set.

        """
        return self._data_handler.perform_data_split(data)

    def get_model_data_for_trace(self, trace):
        """Preprocesses `trace` to retrieve the data used for modelling.

        Parameters
        ----------
        trace: Trace
            A `Trace` object containing the data to be modelled.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the data from `trace` that will be
            used in the modelling process.

        """
        return trace.get_lagged_df(self._lags)

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
        trace_df = self.get_model_data_for_trace(trace)
        X_train, _, X_test, _ = self.split_data(trace_df)
        return self._get_train_and_test_predictions(X_train, X_test)

    def run_model_pipeline_for_trace(self, trace):
        """Runs the model pipeline on `trace`.

        The dataframe containing the data for modeling is obtained from
        `trace`. Then this data is split into features and target for both the
        training and test sets. Next, the model pipeline is run on the
        resulting data to calculate the mean absolute scaled error for the
        training and testing sets, respectively. The number of under
        predictions and the magnitude of the maximum under prediction for the
        test set are also calculated.

        Parameters
        ----------
        trace: Trace
            A `Trace` object containing the data being modelled.
        kwargs: dict
            Arbitrary keyword arguments used to initialize the model.

        Returns
        -------
        ModelResults
            A `ModelResults` object containing the results of building the
            model on `trace`.

        """
        raw_data = self.get_model_data_for_trace(trace)
        X_train, y_train, X_test, y_test = self.split_data(raw_data)
        return self._run_model_pipeline(X_train, y_train, X_test, y_test)

    def plot_trace_vs_prediction(self, trace):
        """Creates a plot of `trace` vs its predictions.

        Parameters
        ----------
        trace: Trace
            The `Trace` being plotted.

        Returns
        -------
        None

        """
        trace_df = self.get_model_data_for_trace(trace)
        title = "Trace {0} vs {1} Predictions".format(
            trace.get_trace_id(), self.get_model_title())
        self._plot_trace_data_vs_predictions(trace_df, title)

    @abstractmethod
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
        pass

    @abstractmethod
    def _run_model_pipeline(self, train_features, train_target,
                            test_features, test_target):
        """Runs the model pipeline on the training and testing data.

        The model is instantiated and then fit on `train_features` and
        `train_target`. Predictions are made on `test_features` and these
        predictions are compared to `test_target` using the mean squared
        error.

        Parameters
        ----------
        train_features: pd.DataFrame
            A pandas DataFrame representing the features for the training set.
        train_target: pd.Series
            A pandas Series representing the target variable for the
            training set.
        test_features: pd.DataFrame
            A pandas Dataframe representing the features for the testing set.
        test_target: pd.Series
            A pandas Series representing the target variable for the testing
            set.

        Returns
        -------
        ModelResults
            A `ModelResults` object containing the results of building the
            model on `trace`.

        """
        pass

    @abstractmethod
    def _plot_trace_data_vs_predictions(self, trace_df, title):
        """Plots the target time series of `trace_df` vs its model prediction.

        The plot of the time series vs its predictions is divided into two
        subplots: one for the training set and another for the testing set.

        Parameters
        ----------
        trace_df: pd.DataFrame
            A pandas DataFrame containing the trace data used for modeling.
        title: str
            A string representing the title of the plot.

        Returns
        -------
        None

        """
        pass
