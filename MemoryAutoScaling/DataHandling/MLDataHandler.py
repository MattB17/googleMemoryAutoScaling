"""The `MLDataHandler` is a class used to pre-process data so that the data
can be used for a machine learning model. It is specifically built for
handling time series data.

"""
from MemoryAutoScaling import utils


class MLDataHandler:
    """Used to pre-process data to be used in a machine learning model.

    The handler is designed for time series data and splits the data so that
    time series models are trained on the first `train_prop` proportion of
    the data and tested on the remaining `1 - train_prop` proportion.

    Parameters
    ----------
    train_prop: float
        A float in [0, 1] representing the proportion of data used in the
        training set.
    feature_vars: list
        A list of strings representing the names of features.
    target_vars: list
        A list of strings representing the names of the target variables.

    Attributes
    ----------
    _train_prop: float
        The proportion of data to be used in the training set.
    _feature_vars: list
        The names of the feature variables.
    _target_vars: list
        The names of the target variables.

    """
    def __init__(self, feature_vars, target_vars,
                 train_prop=0.6, val_prop=0.2):
        self._feature_vars = feature_vars
        self._target_vars = target_vars
        self._train_prop = train_prop
        self._val_prop = val_prop

    def get_target_variables(self):
        """The target variables for the handler.

        Returns
        -------
        list
            A list of strings representing the target variables.

        """
        return self._target_vars

    def get_train_and_test_sets(self, data, tuning=True):
        """Splits `data` into a training and testing set.

        If `tuning` is True, the testing and validation sets are retrieved.
        Otherwise, the testing + validation and testing sets are retrieved.

        Parameters
        ----------
        data: pd.DataFrame
            A pandas DataFrame representing the time series data being split
            into the training and testing sets.
        tuning: bool
            A boolean value indicating whether the data will be used for
            tuning.

        Returns
        -------
        pd.DataFrame, pd.DataFrame
            Two pandas DataFrames where the first dataframe contains the
            records in the training set and the second contains the records
            in the testing set.

        """
        data = data[self._feature_vars + self._target_vars]
        train_thresh, test_thresh = utils.calculate_split_thresholds(
            data, self._train_prop, self._val_prop, tuning)
        return data[:train_thresh], data[train_thresh:test_thresh]

    def perform_data_split(self, data):
        """Splits data into features and targets for the train and test sets.

        Parameters
        ----------
        data: pd.DataFrame
            A pandas DataFrame containing the data on which the split is
            performed.

        Returns
        -------
        pd.Object, pd.Object, pd.Object, pd.Object
            Two pandas objects representing the training set values of the
            features and target variables, respectively. The second pair of
            pandas objects represent the same split for the testing set.

        """
        train_df, test_df = self.get_train_and_test_sets(data)
        y_train, y_test = self._get_train_and_test_targets(train_df, test_df)
        return (train_df[self._feature_vars], y_train,
                test_df[self._feature_vars], y_test)

    def _get_train_and_test_targets(self, train_df, test_df):
        """Retrieves the target data for the training and testing sets.

        Parameters
        ----------
        train_df: pd.DataFrame
            A pandas DataFrame containing the data for the training set.
        test_df: pd.DataFrame
            A pandas DataFrame containing the data for the testing set.

        Returns
        -------
        pd.Object, pd.Object
            Two pandas objects representing the data for the target variables
            for the training and testing sets, respectively.

        """
        if len(self._target_vars) == 1:
            return (train_df[self._target_vars[0]],
                    test_df[self._target_vars[0]])
        return train_df[self._target_vars], test_df[self._target_vars]

    def get_total_spare_for_target(self, trace):
        """The spare amount of the target over the test period for `trace`.

        Parameters
        ----------
        trace: Trace
            The `Trace` object for which the amount of spare units of the
            target variable are computed over the test period.

        Returns
        -------
        float
            A float representing the total amount of spare units of the target
            variable over the test period for `trace`.

        """
        target_ts = trace.get_target_time_series(self._target_vars[0])
        train_cutoff = utils.get_train_cutoff(target_ts, self._train_prop)
        return trace.get_spare_resource_in_window(
            self._target_vars[0], train_cutoff, len(target_ts))
