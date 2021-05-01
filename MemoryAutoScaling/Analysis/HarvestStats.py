"""The `HarvestStats` reports harvesting statistics for a trace based on
its actual values versus predicted values and a buffer.

"""
import numpy as np
from itertools import product
from MemoryAutoScaling import specs, utils


class HarvestStats:
    """Calculates and stores harvest statistics for a trace.

    Parameters
    ----------
    prop_harvested: float
        A float representing the proportion of spare resources harvested.
    prop_violations: float
        A float representing the proportion of violations. That is, the
        proportion of times that the prediction is lower than the actual
        value, even after adding the buffer.

    Attributes
    ----------
    _prop_harvested: float
        The proportion of spare resources harvested.
    _prop_violations: float
        The proportion of predictions that result in violations.

    """
    def __init__(self, prop_harvested, prop_violations):
        self._prop_harvested = prop_harvested
        self._prop_violations = prop_violations

    @classmethod
    def from_predictions(cls, avail_val, actuals, predicteds, buffer_pct):
        """Builds a `HarvestStats` object based on predictions.

        Parameters
        ----------
        avail_val: float
            A float representing the amount of the resource allocated to the
            trace for its duration.
        actuals: np.array
            A numpy array of actual values for the trace, representing the
            percent of the available resource used at each time point.
        predicteds: np.array
            A numpy array of predicted values for the trace, representing the
            percent of the available resource predicted to be used at each
            time point.
        buffer_pct: float
            A non-negative float denoting the percentage of each prediction which
            will act serve as a buffer for the predictions. So
            `(1 + buffer_pct)` is multiplied by the prediction in each period
            to get the prediction for that period.

        Returns
        -------
        HarvestStats
            The `HarvestStats` object in which the proportion harvested and
            the proportion of violations is calculated based on `actuals`,
            `predicteds` and `buffer_pct`.

        """
        prop_harvested, prop_violations = utils.calculate_harvest_stats(
            avail_val, actuals, predicteds, buffer_pct)
        return cls(prop_harvested, prop_violations)

    @classmethod
    def build_null_harvest_stats(cls):
        """Builds a null `HarvestStats` object.

        A null `HarvestStats` object has null values for both the proportion
        harvested and the proportion of violations.

        Returns
        -------
        HarvestStats
            A null `HarvestStats` object.

        """
        return cls(np.nan, np.nan)

    @classmethod
    def get_harvest_stat_columns(cls):
        """The harvest statistic columns.

        Returns
        -------
        list
            A list of strings representing the names of the harvest
            statistics columns.

        """
        return ["prop_harvested", "prop_violations"]

    @classmethod
    def get_harvest_columns_for_buffers(cls):
        """Gets the harvest statistics columns for all buffer percents.

        Returns
        -------
        list
            A list of strings representing the names of the harvest
            statistics columns for all buffer percents.

        """
        buffer_harvest_pairs = product(
            specs.BUFFER_PCTS, cls.get_harvest_stat_columns())
        return ["{0}_{1}".format(harvest_col, buffer_pct)
                for buffer_pct, harvest_col in buffer_harvest_pairs]


    def to_list(self):
        """A list representation of the harvest statistics.

        Returns
        -------
        list
            A list containing the harvest statistics.

        """
        return [self._prop_harvested, self._prop_violations]

    def is_better(self, other_stats):
        """Indicates if the harvest stats are better than `other_stats`.

        The current harvest stats are better than `other_stats` in one of 3
        scenarios: if the proportion harvested is greater and the proportion
        of violations is lower, if the proporition of violations is lower and
        the decrease in violations is at least `specs.HARVEST_WEIGHT` times
        greater than the decrease in proportion harvested, or if the
        proportion harvested is higher and is at least
        `1 / specs.HARVEST_WEIGHT` greater than the increase in violations.

        Parameters
        ----------
        other_stats: HarvestStats
            The `HarvestStats` object to which the current harvest stats are
            compared.

        Returns
        -------
        bool
            True if the current harvest stats are better than
            `other_stats`. Otherwise, False.

        """
        if np.isnan(other_stats._prop_harvested):
            return True
        if np.isnan(self._prop_harvested):
            return False
        return self._prop_harvested > other_stats._prop_harvested
