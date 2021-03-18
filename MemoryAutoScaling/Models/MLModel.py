"""The `MLModel` class is an abstract base class used for all machine
learning models that predict using a time series. It represents an interface
providing the basic framework used by all machine learning models.

"""
from abc import ABC, abstractmethod
from MemoryAutoScaling.DataHandling import MLDataHandler


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

    Attributes
    ----------
    _model_name: str
        The name of the machine learning model.
    _data_handler: MLDataHandler
        The handler used to pre-process data for the model.

    """
    def __init__(self, model_name, data_handler):
        self._model_name = model_name
        self._data_handler = data_handler

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
        return self._data_handler.perform_data_split()
