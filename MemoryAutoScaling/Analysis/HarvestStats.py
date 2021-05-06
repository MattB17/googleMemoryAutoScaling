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
    def from_predictions(cls, trace, predictions, buffer_pct,
                         target_col, pred_start, pred_end):
        """Builds a `HarvestStats` object from the predictions for `trace`.

        `predictions` are a set of predictions of the usage of `target_col`
        of `trace`, based on the aggregation window defined by the trace.
        The predictions are for the period [`pred_start`, `pred_end`]. A
        buffer of `buffer_pct` is applied to each prediction so that the true
        predictions are `1 + buffer_pct` multiplied by `predictions`

        Parameters
        ----------
        trace: Trace
            The `Trace` object for which the predictions were generated.
        predictions: np.array
            A numpy array representing the predictions for which the harvest
            statistics are calculated.
        buffer_pct: float
            A float representing the percent buffer to apply to each
            prediction. That is each prediction is scaled up by
            `1 + buffer_pct`.
        target_col: str
            A string representing the target time series for which the
            predictions were generated.
        pred_start: int
            An integer representing the index of the time series at which
            the predictions start.
        pred_end: int
            An integer representing the index of the time series at which
            the predictions end.

        Returns
        -------
        HarvestStats
            The `HarvestStats` object in which the proportion harvested and
            the proportion of violations is calculated based on the
            predictions for `target_col` of `trace`.

        """
        actuals = trace.get_target_time_series(target_col)[pred_start:pred_end]
        allocated_amt = trace.get_amount_allocated_for_target(target_col)
        harvest_amt, num_violations = utils.calculate_harvest_stats(
            allocated_amt, actuals, predictions, buffer_pct)
        total_spare = trace.get_spare_resource_in_window(
            target_col, pred_start, pred_end)
        harvest_amt *= trace.get_aggregation_window()
        return cls(harvest_amt / total_spare,
                   num_violations / len(predictions))

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
