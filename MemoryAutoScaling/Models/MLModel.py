"""The `MLModel` class is an abstract base class used for all machine
learning models that predict using a time series. It represents an interface
providing the basic framework used by all machine learning models.

"""
from abc import ABC, abstractmethod


class MLModel(ABC):
    """Used to predict future values of a time series.

    Parameters
    ----------
    model_name: str
        A string representing the name of the machine learning model.

    Attributes
    ----------
    _model_name: str
        The name of the machine learning model.
    
    """
    def __init__(self, model_name):
        self._model_name = model_name
