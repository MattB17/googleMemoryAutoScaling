import pytest
import numpy as np
import pandas as pd
from MemoryAutoScaling.utils import cap_and_clean_values

@pytest.fixture(scope="function")
def mock_df():
    return pd.DataFrame({"A": [0.7, 0.6, 0.8],
                         "B": [1.3, 0.9, 1.1],
                         "C": [1.1, np.nan, 0.1]})


def test_with_empty_dataframe():
    df = pd.DataFrame({"A": [], "B": []})
    result = cap_and_clean_values(df, "A", 1.0)
    assert result.tolist() == []


def test_with_one_element_dataframe():
    df = pd.DataFrame({"A": [1.1], "B": [0.7]})
    result = cap_and_clean_values(df, "A", 1.0)
    assert result.tolist() == [1.0]


def test_with_no_cap(mock_df):
    result = cap_and_clean_values(mock_df, "A", 1.0)
    assert result.tolist() == [0.7, 0.6, 0.8]


def test_with_cap(mock_df):
    result = cap_and_clean_values(mock_df, "B", 1.0)
    assert result.tolist() == [1.0, 0.9, 1.0]


def test_with_nan(mock_df):
    result = cap_and_clean_values(mock_df, "C", 1.0)
    assert result.tolist() == [1.0, 0.0, 0.1]
