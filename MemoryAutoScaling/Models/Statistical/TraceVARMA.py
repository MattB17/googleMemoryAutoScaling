"""The `TraceVARMA` class is used to construct a predictive model based on
VARMA. VARMA models are used for multivariate time series predictions.
They have two terms: `p` and `q`. `p` is the autoregressive component, that
is the number of previous values used to predict the current value. `q` is
the moving average component, referring to the number of lagged forecast
errors used in the model.

"""
from MemoryAutoScaling import parallel, utils
from MemoryAutoScaling.Models.Statistical import StatisticalModel
from statsmodels.tsa.statespace.varmax import VARMAX


class TraceVARMA(StatisticalModel):
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
    def __init__(self, model_vars, p, q, train_prop=0.7):
        super().__init__(train_prop)
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

    def get_model_vars(self):
        """The variables being modelled.

        Returns
        -------
        list
            A list of strings representing the names of variables
            being modeled.

        """
        return self._model_vars

    def get_model_title(self):
        """The title for the model.

        Returns
        -------
        str
            A string representing the title for the model.

        """
        return "Trace VARMA {}".format(self.get_params())

    def split_data(self, model_data):
        """Splits `model_data` into training and testing sets.

        Parameters
        ----------
        model_data: pd.DataFrame
            A pandas DataFrame containing the time series that are being
            modeled, with a separate column for each time series that is
            being modeled by the multivariate model.

        Returns
        -------
        pd.DataFrame, pd.DataFrame
            Two pandas DataFrames representing the records in the train and
            test sets respectively.

        """
        train_cutoff = utils.get_train_cutoff(model_data, self._train_prop)
        model_data = model_data.reset_index(drop=True)
        return model_data[:train_cutoff], model_data[train_cutoff:]

    def get_model_data_for_trace(self, trace):
        """Retrieves the multivariate data for modeling from `trace`.

        Parameters
        ----------
        trace: Trace
            The `Trace` object being modeled.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame representing the multivariate data being
            modeled. It has one column for each time series being modeled.

        """
        return trace.get_trace_df()[self._model_vars]

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
        dict
            A dictionary of model results. The keys are strings representing
            the modeled variables and the corresponding value is a
            `ModelResults` object representing the results for that variable.

        """
        trace_df = self.get_model_data_for_trace(trace)
        train_df, test_df = self.split_data(trace_df)
        self._fit(train_df)
        preds_train, preds_test = self._get_predictions(len(test_df))
        return parallel.get_multivariate_model_results(
            self.get_params(), train_df, preds_train, test_df,
            preds_test, self._model_vars)

    def plot_trace_vs_prediction(self, trace):
        """Creates a plot of `trace` vs its prediction.

        Each row contains two subplots, for the training and testing sets,
        respectively. There is a row for each of the model variables.

        Parameters
        ----------
        trace: Trace
            The `Trace` object for which actual and predicted values are
            plotted.

        Returns
        -------
        None

        """
        var_count = len(self._model_vars)
        fig, axes = plt.subplots(var_count, 2, figsize=(20, 10*var_count))
        trace_df = self.get_model_data_for_trace(trace)
        train_df, test_df = self.split_data(trace_df)
        preds_train, preds_test = self._get_predictions(len(test_df))
        utils.plot_multivariate_train_and_test_predictions(
            train_df, preds_train, test_df, preds_test,
            axes, self._model_vars, self.get_plot_title())

    def _fit(self, train_data):
        """Fits the model based on training data `train_data`.

        Parameters
        ----------
        train_data: pd.DataFrame
            A pandas DataFrame representing the data used for training.

        Returns
        -------
        None

        """
        model = VARMAX(train_data, order=self.get_params())
        self._model = model.fit(disp=False)
