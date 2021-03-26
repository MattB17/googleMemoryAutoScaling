"""The `TimeSeriesSVM` class builds a Support Vector Machine model to predict
future time points for a time series.

"""
from MemoryAutoScaling.Models import MLModel
from sklearn.svm import SVR


class TimeSeriesSVM(MLModel):
    """A Support Vector Machine model for time series data.

    Parameters
    ----------
    data_handler: MLDataHandler
        An `MLDataHandler` used to pre-process data for the model.
    lags: list
        A list of integers representing the lags to use for time series
        features to the model. That is, for any time series used as a feature
        to the model, the model will use the value of the time series lagged
        by the time points in `lags` when predicting for the target variable.
    reg_val: float
        A float specifying the degree of regularization for the model.

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
        Indicates if the model has been fit to training data.
    _reg_val: float
        The regularization parameter for the model.

    """
    def __init__(self, data_handler, lags, reg_val):
        super().__init__("TimeSeriesSVM", data_handler, lags)
        self._reg_val = reg_val

    def initialize(self, **kwargs):
        """Initializes the support vector machine model.

        Parameters
        ----------
        kwargs: dict
            Arbitrary keyword arguments used in initialization.

        Returns
        -------
        None

        """
        super().initialize()
        self._model = SVR(**kwargs)

    def get_train_and_test_predictions(self, train_features, test_features):
        """Retrieves predictions for the training and testing sets.

        Parameters
        ----------
        train_features: pd.DataFrame
            A pandas DataFrame representing the values for the features of the
            training set.
        test_features: pd.DataFrame
            A pandas DataFrame representing the values for the features of the
            testing set.

        Returns
        -------
        np.array, np.array
            Two numpy arrays representing the predictions fo the training and
            testing sets respectively.

        """
        train_preds = self.get_predictions(train_features)
        test_preds = self.get_predictions(test_features)
        return train_preds, test_preds
