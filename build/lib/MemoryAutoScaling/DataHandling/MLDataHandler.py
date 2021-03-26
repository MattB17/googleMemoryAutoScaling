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
    target_var: str
        A string representing the name of the target variable.

    Attributes
    ----------
    _train_prop: float
        The proportion of data to be used in the training set.
    _feature_vars: list
        The names of the feature variables.
    _target_var: list
        The name of the target variable.

    """
    def __init__(self, train_prop, feature_vars, target_var):
        self._train_prop = train_prop
        self._feature_vars = feature_vars
        self._target_var = target_var

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
        data = data[self._feature_vars + [self._target_var]]
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
        pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
            A pandas DataFrame and Series representing the training set values
            of the features and target variable, respectively. The second
            DataFrame and Series represent the same split for the testing set.

        """
        train_df, test_df = self.get_train_and_test_sets(data)
        return (train_df[self._feature_vars], train_df[self._target_var],
                test_df[self._feature_vars], test_df[self._target_var])
