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
    def __init__(self, train_prop, feature_vars, target_vars):
        self._train_prop = train_prop
        self._feature_vars = feature_vars
        self._target_vars = target_vars

    def get_target_variables(self):
        """The target variables for the handler.

        Returns
        -------
        list
            A list of strings representing the target variables.

        """
        return self._target_vars

    def get_train_and_test_sets(self, data):
        """Splits `data` into a training and testing set.

        Parameters
        ----------
        data: pd.DataFrame
            A pandas DataFrame representing the time series data being split
            into the training and testing sets.

        Returns
        -------
        pd.DataFrame, pd.DataFrame
            Two pandas DataFrames where the first dataframe contains the
            records in the training set and the second contains the records
            in the testing set.

        """
        data = data[self._feature_vars + self._target_vars]
        train_cutoff = utils.get_train_cutoff(data, self._train_prop)
        return data[:train_cutoff], data[train_cutoff:]

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
