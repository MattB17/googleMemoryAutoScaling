import sys
import random
import numpy as np
import pandas as pd
from MemoryAutoScaling import TraceGenerator

random.seed(42)
np.random.seed(42)

if __name__ == "__main__":
    trace_length = int(sys.argv[1])
    file_path = sys.argv[2]
    generator = TraceGenerator(trace_length, 50, 5.74, 2.36)
    constant_traces = generator.generate_constant_traces(350)
    periodic_traces = generator.generate_periodic_traces(600, 10, 2, 30, 4.5)
    unpredictable_traces = generator.generate_unpredictable_traces(
        50, 29, 1.7, 85, 7.9)
    df = pd.DataFrame(
        constant_traces + periodic_traces + unpredictable_traces,
        columns=["t{}".format(pos) for pos in range(trace_length)])
    df.to_csv(file_path, sep=",", index=False)
