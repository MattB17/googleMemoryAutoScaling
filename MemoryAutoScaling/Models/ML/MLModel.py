"""The `MLModel` class is an abstract base class used for all machine
learning models that predict using a time series. It represents an interface
providing the basic framework used by all machine learning models.

"""
from abc import abstractmethod
from MemoryAutoScaling import plotting, utils
from MemoryAutoScaling.Models.ML import MLBase
from MemoryAutoScaling.Analysis import ModelResults
from MemoryAutoScaling.DataHandling import MLDataHandler



class MLModel(MLBase):
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
        super().__init__(model_name, data_handler, lags)

    def get_target_variable(self):
        """The target variable for the model.

        Returns
        -------
        str
            A string indicating the target variable for the model.

        """
        return self._data_handler.get_target_variables()[0]

    def get_available_resource_data(self, trace, tuning=True):
        """A time series of the available resource for `trace`.

        The time series is restricted to the evaluation interval specified
        by `tuning`. If `tuning` is True then the model is being tuned so the
        time series is restricted to the validation set. Otherwise, it is
        restricted to the testing set.

        Parameters
        ----------
        trace: Trace
            The `Trace` object from which the available resource numbers are
            retrieved.
        tuning: bool
            A boolean value indicating whether or not the model is being
            tuned.

        Returns
        -------
        np.array
            A numpy array representing the amount of the resource available
            for each time point in the evaluation window specified by `tuning`.

        """
        total_avail_ts = trace.get_target_availability_time_series(
            self.get_target_variable())[max(self._lags):]
        _, avail_ts = self._data_handler.split_time_series_data(
            total_avail_ts, tuning)
        return avail_ts

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

    def _is_initialized(self):
        """Indicates if the model has been initialized.

        Returns
        -------
        bool
            True if the model has been initialized, otherwise False.

        """
        return self._model is not None

    @abstractmethod
    def _initialize(self):
        """Initializes the model.

        Returns
        -------
        None

        """
        self._is_fit = False

    def _fit(self, train_features, train_target):
        """Fits the model based on `train_features` and `train_target`.

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
        if self._is_initialized():
            self._model.fit(train_features, train_target)
            self._is_fit = True

    def _get_predictions(self, input_features):
        """Retrieves model predictions for `input_features`.

        Parameters
        ----------
        input_features: pd.DataFrame
            A pandas DataFrame representing the input features.

        Returns
        -------
        pd.Series
            A pandas Series representing the predictions based on
            `input_features`.

        """
        if self._is_fit:
            return self._model.predict(input_features)

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
        train_preds = self._get_predictions(train_features)
        test_preds = self._get_predictions(test_features)
        return train_preds, test_preds

    def _run_model_pipeline(self, avail_data, X_train, train_target,
                            X_test, test_target, total_spare):
        """Runs the model pipeline on the training and testing data.

        The model is instantiated and then fit on `X_train` and
        `train_target`. Predictions are made on `X_test` and these
        predictions are compared to `test_target` using the mean squared
        error.

        Parameters
        ----------
        avail_data: np.array
            A numpy array representing a time series of the availability of
            the target resource.
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
        total_spare: float
            A float representing the total spare amount of the target variable
            over the test period.

        Returns
        -------
        ModelResults
            A `ModelResults` object containing the results of building the
            model on `trace`.

        """
        self._initialize()
        self._fit(X_train, train_target)
        train_preds, test_preds = self._get_train_and_test_predictions(
            X_train, X_test)
        return ModelResults.from_data(
            self.get_params(), avail_data, train_target,
            train_preds, test_target, test_preds, total_spare)

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
