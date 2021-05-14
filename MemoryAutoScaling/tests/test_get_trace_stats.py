import numpy as np
from MemoryAutoScaling.utils import get_trace_stats


def test_with_empty_array():
    assert get_trace_stats(np.array([])) == {}


def test_with_1_element_array():
    val = 1.3
    return_dict = get_trace_stats(np.array([val]))
    assert round(return_dict['max'], 2) == 1.30
    assert round(return_dict['avg'], 2) == 1.30
    assert round(return_dict['median'], 2) == 1.30
    assert round(return_dict['std'], 2) == 0.00
    assert round(return_dict['range'], 2) == 0.00


def test_with_larger_array():
    return_dict = get_trace_stats(np.array([2.1, 2.2, 2.4, 2.9]))
    assert round(return_dict['max'], 2) == 2.90
    assert round(return_dict['avg'], 2) == 2.40
    assert round(return_dict['median'], 2) == 2.30
    assert round(return_dict['std'], 2) == 0.31
    assert round(return_dict['range'], 2) == 0.80
