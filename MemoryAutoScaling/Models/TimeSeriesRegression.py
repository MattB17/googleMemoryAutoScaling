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

    def get_modelling_data_from_trace(self, trace):
        """Preprocesses `trace` to retrieve the data used for modelling.

        `trace` is processed to retrieve a DataFrame containing the target
        variable as well as the lagged values of all feature variables at a
        lag of `lag`.

        Parameters
        ----------
        trace: Trace
            The `Trace` object from which the modelling data is retrieved.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the data to be modelled.

        """
        return trace.get_lagged_df(self._lag)
