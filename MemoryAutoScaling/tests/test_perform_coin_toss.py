from unittest.mock import patch
from MemoryAutoScaling.utils import perform_coin_toss


BIN_STR = "numpy.random.binomial"


def test_with_low_probability():
    with patch(BIN_STR, return_value=0) as mock_binomial:
        assert perform_coin_toss(0.1) == 0
    mock_binomial.assert_called_once_with(1, 0.1)


def test_with_high_probability():
    with patch(BIN_STR, return_value=1) as mock_binomial:
        assert perform_coin_toss(0.9) == 1
    mock_binomial.assert_called_once_with(1, 0.9) 
