"""The `TimeSeriesRegression` class builds a linear regression model to
predict future time points for a time series.

"""
from MemoryAutoScaling.Models import MLModel
from sklearn.linear_model import LinearRegression


class TimeSeriesRegression(MLModel):
    """A LinearRegression model for time series data.

    Parameters
    ----------
    data_handler: MLDataHandler
        An `MLDataHandler` used to pre-process data for the model.
    lag: int
        An integer representing the lag to use for time series features to
        the model. That is, for any time series used as a feature to the
        model, the values of that time series lagged by `lag` will be the
        feature used in the regression.

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
    def __init__(self, data_handler, lag):
        super().__init__("TimeSeriesRegression", data_handler, lag)

    def initialize(self, **kwargs):
        """Initializes the linear regression model.

        Parameters
        ----------
        kwargs: dict
            Arbitrary keyword arguments used in initialization.

        Returns
        -------
        None

        """
        super().initialize()
        self._model = LinearRegression(**kwargs)

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
            Two numpy arrays representing the predictions for the training and
            testing sets respectively.


        """
        train_preds = self.get_predictions(train_features)
        test_preds = self.get_predictions(test_features)
        return train_preds, test_preds
