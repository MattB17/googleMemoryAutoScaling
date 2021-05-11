import numpy as np
from unittest.mock import patch, MagicMock, call
from MemoryAutoScaling.utils import cap_train_and_test_predictions


CAP_STR = "MemoryAutoScaling.utils.cap_predictions_at_1"


def test_with_empty_preds():
    initial_train_preds = np.array([])
    initial_test_preds = np.array([])
    with patch(CAP_STR, side_effect=(np.array([]), np.array([]))) as mock_cap:
        train_preds, test_preds = cap_train_and_test_predictions(
            initial_train_preds, initial_test_preds)
    assert len(train_preds) == 0
    assert len(test_preds) == 0
    assert mock_cap.call_count == 2
    cap_calls = [call(initial_train_preds), call(initial_test_preds)]
    mock_cap.assert_has_calls(cap_calls)


def test_with_small_preds():
    initial_train_preds = np.array([0.7, 1.3, 0.8])
    initial_test_preds = np.array([0.9])
    capped_train_preds = np.array([0.7, 1.0, 0.8])
    capped_test_preds = np.array([0.9])
    with patch(CAP_STR,
        side_effect=(capped_train_preds, capped_test_preds)) as mock_cap:
        train_preds, test_preds = cap_train_and_test_predictions(
            initial_train_preds, initial_test_preds)
    assert train_preds.tolist() == capped_train_preds.tolist()
    assert test_preds.tolist() == capped_test_preds.tolist()
    assert mock_cap.call_count == 2
    cap_calls = [call(initial_train_preds), call(initial_test_preds)]
    mock_cap.assert_has_calls(cap_calls)


def test_with_large_preds():
    initial_train_preds = np.array([0.35, 0.67, 0.43, 0.51, 0.6, 0.7, 0.85])
    initial_test_preds = np.array([1.1, 0.7, 1.3])
    capped_train_preds = np.array([0.35, 0.67, 0.43, 0.51, 0.6, 0.7, 0.85])
    capped_test_preds = np.array([1.0, 0.7, 1.0])
    with patch(CAP_STR,
        side_effect=(capped_train_preds, capped_test_preds)) as mock_cap:
        train_preds, test_preds = cap_train_and_test_predictions(
            initial_train_preds, initial_test_preds)
    assert train_preds.tolist() == capped_train_preds.tolist()
    assert test_preds.tolist() == capped_test_preds.tolist()
    assert mock_cap.call_count == 2
    cap_calls = [call(initial_train_preds), call(initial_test_preds)]
    mock_cap.assert_has_calls(cap_calls)
