"""
Implementations of different score aggregation functions.
"""


import numpy as np
from numpy import ndarray

from ..utils.registry import register


class BaseAggregator:
    def __call__(self, array: ndarray, axis: int) -> ndarray:
        """
        Score aggregation logic.

        Args:
            array (ndarray): Score array that needs to be aggregated.
            axis (int): Dimension along which to aggregate scores.
        """
        raise NotImplementedError


@register("aggregator")
class DoNothingAggregator(BaseAggregator):
    """
    Raise an error when called.
    """
    pass


@register("aggregator")
class SumAggregator(BaseAggregator):
    def __call__(self, array: ndarray, axis: int) -> ndarray:
        if array.size == 0:
            return np.nan
        return array.sum(axis=axis)


@register("aggregator")
class MeanAggregator(BaseAggregator):
    def __call__(self, array: ndarray, axis: int) -> ndarray:
        if array.size == 0:
            return np.nan
        return array.mean(axis=axis)


@register("aggregator")
class MaxAggregator(BaseAggregator):
    def __call__(self, array: ndarray, axis: int) -> ndarray:
        if array.size == 0:
            return np.nan
        return array.max(axis=axis)


@register("aggregator")
class MinAggregator(BaseAggregator):
    def __call__(self, array: ndarray, axis: int) -> ndarray:
        if array.size == 0:
            return np.nan
        return array.min(axis=axis)


@register("aggregator")
class MedianAggregator(BaseAggregator):
    def __call__(self, array: ndarray, axis: int) -> ndarray:
        if array.size == 0:
            return np.nan
        return np.median(array, axis=axis)


@register("aggregator")
class TruncatedMeanAggregator(BaseAggregator):
    """
    Truncated mean score aggregator.

    Args:
        start : float
            Must be in range [0.0, 1.0]. Marks the beginning of the truncated array.
        end : float
            Must be in range [0.0, 1.0]. Marks the end of the truncated array.
    """
    def __init__(self, start: float, end: float):
        assert 0.0 <= start < end <= 1.0

        self.start = start
        self.end = end

    def __call__(self, array: ndarray, axis: int) -> ndarray:
        if array.size == 0:
            return np.nan

        length = array.shape[axis]
        start = round(length * self.start)
        end = round(length * self.end)

        if start == end:
            return np.nan
        if start > end:
            raise ValueError(f"Got invalid values: start: {start}, end: {end}.")

        # Sort array and select the right portion
        array = np.sort(array, axis=axis)
        indices = np.arange(start=start, stop=end)
        array = np.take(array, indices, axis=axis)

        array = array.mean(axis=axis)
        return array
