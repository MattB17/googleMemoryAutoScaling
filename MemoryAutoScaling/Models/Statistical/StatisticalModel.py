"""The `StatisticalModel` class is an abstract base class used for all
statistical models used for time series prediction. It represents the
interface used by these models.

"""
from abc import abstractmethod
from MemoryAutoScaling import utils
from MemoryAutoScaling.Models import TraceModel


class StatisticalModel(TraceModel):
    """A statistical model for a data trace.

    Parameters
    ----------
    train_prop: float
        A float in the range [0, 1] representing the proportion of data
        in the training set. The default value is 0.7.

    Attributes
    ----------
    _train_prop: float
        The proportion of data in the training set.
    _model: Object
        The underlying model being fit to the trace data.

    """
    def __init__(self, train_prop=0.7):
        self._train_prop = train_prop
        self._model = None

    @abstractmethod
    def get_params(self):
        """Returns the parameters of the model.

        Returns
        -------
        tuple
            A tuple containing the model parameters.

        """
        pass
