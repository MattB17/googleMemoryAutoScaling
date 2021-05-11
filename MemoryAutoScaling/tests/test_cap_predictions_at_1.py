import numpy as np
from MemoryAutoScaling.utils import cap_predictions_at_1


def test_empty_array():
    result = cap_predictions_at_1(np.array([]))
    assert len(result) == 0


def test_array_of_size_1():
    assert cap_predictions_at_1(np.array([1.3])) == np.array([1.0])


def test_array_with_multi_elements():
    result = cap_predictions_at_1(np.array([0.7, 1.47, 0.65]))
    assert result.tolist() == [0.7, 1.0, 0.65]
