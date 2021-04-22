"""The `HarvestStats` reports harvesting statistics for a trace based on
its actual values versus predicted values and a buffer.

"""
from MemoryAutoScaling import utils


class HarvestStats:
    """Calculates and stores harvest statistics for a trace.

    Parameters
    ----------
    actuals: np.array
        A numpy array of actual values for the trace.
    predicteds: np.array
        A numpy array of predicted values for the trace.
    buffer_pct: float
        A non-negative float denoting the percentage of each prediction which
        will act serve as a buffer for the predictions. So `(1 + buffer_pct)`
        is multiplied by the prediction in each period to get the prediction
        for that period.

    Attributes
    ----------
    _prop_harvested: float
        Represents the proportion of actual available memory that is
        successfully harvested for the trace.
    _prop_violations: float
        The proportion of predictions that result in violations. That is,
        the proportion of times that the prediction is lower than the actual
        value, even after adding the buffer.

    """
    def __init__(self, actuals, predicteds, buffer_pct):
        prop_harvested, prop_violations = utils.calculate_harvest_stats(
            list(actuals), list(predicteds), buffer_pct)
        self._prop_harvested = prop_harvested
        self._prop_violations = prop_violations

    @classmethod
    def get_harvest_stat_columns(cls):
        """The harvest statistic columns.

        Returns
        -------
        list
            A list of strings representing the names of the harvest
            statistics.

        """
        return ["prop_harvested", "prop_violations"]

    def to_list(self):
        """A list representation of the harvest statistics.

        Returns
        -------
        list
            A list containing the harvest statistics.

        """
        return [self._prop_harvested, self._prop_violations]

    def is_better(self, other_harvest_stats):
        """Indicates if the stats are better than `other_harvest_stats`.

        The current harvest stats are better than `other_harvest_stats` if
        the proportion harvested is higher for the current harvest stats
        compare to `other_harvest_stats`.

        Parameters
        ----------
        other_harvest_stats: HarvestStats
            The `HarvestStats` object to which the current harvest stats are
            compared.

        Returns
        -------
        bool
            True if the current harvest stats are better than
            `other_harvest_stats`. Otherwise, False.

        """
        return self._prop_harvested > other_harvest_stats._prop_harvested
