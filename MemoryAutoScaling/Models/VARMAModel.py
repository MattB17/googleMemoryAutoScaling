"""The `VARMAModel` class is used to construct a predictive model based on
VARMA. VARMA models are used for multivariate time series predictions.
They have two terms: `p` and `q`. `p` is the autoregressive component, that
is the number of previous values used to predict the current value. `q` is
the moving average component, referring to the number of lagged forecast
errors used in the model.

"""
from MemoryAutoScaling import utils
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.varmax import VARMAX


class VARMAModel:
    """A VARMA model for time series data.

    Parameters
    ----------
    train_prop: float
        A float in the range [0, 1] representing the proportion of data
        in the training set.
    model_vars: list
        A list of strings representing the names of variables being modeled.
    p: int
        An integer representing the autoregressive component of the model.
    q: int
        An integer representing the moving average component of the model.

    Attributes
    ----------
    _train_prop: float
        The proportion of data in the training set.
    _model: Object
        The underlying machine learning model being fit.
    _model_vars: list
        The variables being modeled.
    _p: int
        The autoregressive component of the model.
    _q: int
        The moving average component of the model.

    """
    def __init__(self, train_prop, model_vars, p, q):
        self._train_prop = train_prop
        self._model = None
        self._model_vars = model_vars
        self._p = p
        self._q = q

    def get_params(self):
        """Returns the order for the ARIMA model.

        The order is the three element tuple `(p, q)` representing the
        autoregressive component and the moving average component,
        respectively.

        Returns
        -------
        tuple
            A two element tuple of integers representing the order of the
            VARMA model.

        """
        return self._p, self._q

    def split_data(self, trace_df):
        """Splits `trace_df` into training and testing data sets.

        Parameters
        ----------
        trace_df: pd.DataFrame
            A pandas DataFrame containing the data for the time series being
            modeled.

        Returns
        -------
        pd.DataFrame, pd.DataFrame
            Two pandas DataFrames representing the records in the train and
            test sets respectively.

        """
        train_cutoff = utils.get_train_cutoff(trace_df, self._train_prop)
        trace_df = trace_df.reset_index(drop=True)
        return trace_df[:train_cutoff], trace_df[train_cutoff:]

    def fit(self, trace_df):
            """Fits the model based on training data, `train_df`.

            Parameters
            ----------
            train_df: pd.DataFrame
                A pandas DataFrame representing the data used for training.

            Returns
            -------
            None

            """
            model = VARMAX(train_df, order=self.get_params())
            self._model = model.fit(disp=False)

    def get_predictions(self, test_df):
        """Retrieves model predictions for `test_df`.

        Parameters
        ----------
        test_df: pd.DataFrame
            A pandas DataFrame representing the data used for testing.

        Returns
        -------
        VARMAX.prediction
            A `VARMAX.prediction` object containing the predictions
            for the time series up to the end of the testing period.

        """
        forecast_len = len(test_df)
        return self._model.get_prediction(
            end=self._model.nobs + forecast_len - 1)

    def run_model_pipeline(self, train_df, test_df):
        """Runs the model pipeline on `train_df` and `test_df`.

        The model is instantiated and fit based on `train_df`. Then
        predictions are obtained for the test time frame and these
        predictions are compared to `test_df` using the MAPE.

        Parameters
        ----------
        train_df: pd.DataFrame
            A pandas DataFrame representing the proportion of the data
            corresponding to the training set.
        test_df: pd.DataFrame
            A pandas DataFrame representing the proportion of the data
            corresponding to the testing set.

        Returns
        -------
        VARMAX.prediction, list, list
            A `VARMAX.prediction` object containing the predictions for
            the time series up to the end of the testing period and two
            lists corresponding to the MAPEs of the predictions for the
            training and testing sets, respectively, for all of variables
            being modelled.

        """
        self.fit(train_df)
        preds = self.get_predictions(test_df)
        n_forecast = len(test_df)
        train_mapes = []
        test_mapes = []
        for col_name in self._model_vars:
            mean_preds = utils.impute_for_time_series(
                preds.predicted_mean[col_name], 0)
            train_mapes.append(mean_squared_error(
                train_df[col_name], mean_preds[:-n_forecast]))
            test_mapes.append(mean_squared_error(
                test_df[col_name], mean_preds[-n_forecast:]))
        return preds, train_mapes, test_mapes

    def run_model_pipeline_for_trace(self, data_trace):
        """Runs the model pipeline on `data_trace`.

        `data_trace` is first split into a training and testing set and then
        the model pipeline is run on these sets.

        Parameters
        ----------
        data_trace: Trace
            A `Trace` object representing the time series being modelled.

        Returns
        -------
        SARIMAX.prediction, float, float
            A `SARIMAX.prediction` object containing the predictions for
            the time series up to the end of the testing period and two
            floats corresponding to the MAPE of the predictions for the
            training and testing sets respectively.

        """
        trace_df = data_trace.get_trace_df[self._model_vars]
        train_df, test_df = self.split_data(trace_df_)
        return self.run_model_pipeline(train_df, test_df)
