"""The `MLModel` class is an abstract base class used for all machine
learning models that predict using a time series. It represents an interface
providing the basic framework used by all machine learning models.

"""
from abc import ABC, abstractmethod
from MemoryAutoScaling.DataHandling import MLDataHandler
from sklearn.metrics import mean_squared_error


class MLModel(ABC):
    """Used to predict future values of a time series.

    The model has an `MLDataHandler`, `data_handler`, to ensure that all
    data handled by the model is handled consistently.

    Parameters
    ----------
    model_name: str
        A string representing the name of the machine learning model.
    data_handler: MLDataHandler
        An `MLDataHandler` used to pre-process data for the model.
    lag: int
        An integer representing the lag to use for time series features to
        the model. That is, for any time series used as a feature to the
        model, the model will use the value of the time series lagged by
        `lag` when predicting for the target variable.

    Attributes
    ----------
    _model_name: str
        The name of the machine learning model.
    _data_handler: MLDataHandler
        The handler used to pre-process data for the model.
    _model: Object
        The underlying machine learning model being fit.
    _is_fit: bool
        Indicates if the model has been fitted to training data.
    _lag: int
        The lag used for time series features to the model.

    """
    def __init__(self, model_name, data_handler, lag):
        self._model_name = model_name
        self._data_handler = data_handler
        self._lag = lag
        self._model = None
        self._is_fit = False

    def is_initialized(self):
        """Indicates if the model has been initialized.

        Returns
        -------
        bool
            True if the model has been initialized, otherwise False.

        """
        return self._model is not None

    @abstractmethod
    def initialize(self, **kwargs):
        """Initializes the model.

        Parameters
        ----------
        kwargs: dict
            A dictionary of arbitrary keyword parameters which are passed
            to the model at initialization.

        Returns
        -------
        None

        """
        self._is_fit = False

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

    def fit(self, train_features, train_target):
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
        if self.is_initialized():
            self._model.fit(train_features, train_target)
            self._is_fit = True

    def get_predictions(self, test_features):
        """Retrieves model predictions for `test_features`.

        Parameters
        ----------
        test_features: pd.DataFrame
            A pandas DataFrame representing the testing features.

        Returns
        -------
        None

        """
        if self._is_fit:
            return self._model.predict(test_features)

    @abstractmethod
    def get_modelling_data_from_trace(self, trace):
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
        pass

    def run_model_pipeline(self, train_features, train_target,
                           test_features, test_target, **kwargs):
        """Runs the model pipeline on the training and testing data.

        The model is instantiated with `kwargs` and then fit on
        `train_features` and `train_target`. Predictions are made on
        `test_features` and these predictions are compared to `test_target`
        using the mean squared error.

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
        pd.Series, float, float
            A pandas Series representing the predictions for the testing set
            and the MSEs for the model predictions versus the training and
            testing sets.

        """
        self.initialize(**kwargs)
        self.fit(train_features, train_target)
        preds = self.get_predictions(test_features)
        train_mse = mean_squared_error(
            train_target, self.get_predictions(train_features))
        test_mse = mean_squared_error(test_target, preds)
        return preds, train_mse, test_mse

    def run_model_pipeline_on_trace(self, trace, **kwargs):
        """Runs the model pipeline on `trace`.

        The dataframe containing the data for modeling is obtained from
        `trace`. Then this data is split into features and target for both the
        training and test sets. Next, the model pipeline is run on the
        resulting data, with the model being initialized by parameters
        specified in **kwargs.

        Parameters
        ----------
        trace: Trace
            A `Trace` object containing the data being modelled.
        kwargs: dict
            Arbitrary keyword arguments used to initialize the model.

        Returns
        -------
        pd.Series, float, float
            A pandas Series representing the predictions for the testing set
            and the MSEs for the model predictions versus the training and
            testing sets.

        """
        raw_data = self.get_modelling_data_from_trace(trace)
        X_train, y_train, X_test, y_test = self.split_data(raw_data)
        return self.run_model_pipeline(
            X_train, y_train, X_test, y_test, **kwargs)
