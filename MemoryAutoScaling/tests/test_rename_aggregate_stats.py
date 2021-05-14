from MemoryAutoScaling.utils import rename_aggregate_stats


def test_with_no_stats():
    agg_stats = {'avg': [], 'max': [], 'median': [], 'range': [], 'std': []}
    renamed_stats = rename_aggregate_stats(agg_stats, "memory_usage", True)
    assert renamed_stats == {'memory_usage_avg': [],
                             'memory_usage_ts': [],
                             'memory_usage_median': [],
                             'memory_usage_range': [],
                             'memory_usage_std': []}


def test_with_one_period_stats():
    agg_stats = {'max': [4.50], 'avg': [2.67], 'median': [2.60],
                 'range': [3.20], 'std': [1.03]}
    renamed_stats = rename_aggregate_stats(agg_stats, "cpu_usage", False)
    assert renamed_stats == {'cpu_usage_ts': [2.67],
                             'cpu_usage_max': [4.50],
                             'cpu_usage_median': [2.60],
                             'cpu_usage_range': [3.20],
                             'cpu_usage_std': [1.03]}


def test_with_multi_period_stats():
    agg_stats = {'max': [1.50, 2.70, 3.70],
                 'avg': [1.40, 2.50, 3.50],
                 'median': [1.40, 2.50, 3.50],
                 'range': [0.20, 0.40, 0.40],
                 'std': [0.08, 0.16, 0.16]}
    renamed_stats = rename_aggregate_stats(agg_stats, "cpu_usage", True)
    assert renamed_stats == {'cpu_usage_ts': [1.50, 2.70, 3.70],
                             'cpu_usage_avg': [1.40, 2.50, 3.50],
                             'cpu_usage_median': [1.40, 2.50, 3.50],
                             'cpu_usage_range': [0.20, 0.40, 0.40],
                             'cpu_usage_std': [0.08, 0.16, 0.16]}
