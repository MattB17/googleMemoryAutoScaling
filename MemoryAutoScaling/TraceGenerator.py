"""The `TraceGenerator` class is used to generate data traces for analysis.
It has several methods for generating different classes of traces
corresponding to different usage patterns.

"""
import numpy as np
import random


class TraceGenerator:
    """Generates data trace for analysis.

    Parameters
    ----------
    trace_length: int
        An integer representing the number of time points for the generated
        traces.
    mu: float
        A float representing the average for traces. This is the base average
        for all traces and then some amount of noise and or trend is added to
        this value.
    sigma: float
        A float representing the standard deviation for traces. This represents
        the base standard deviation of traces before noise or trend are added.
    noise_amp: float
        A float representing the base standard deviation of noise added to
        each trace.

    Attributes
    ----------
    _trace_length: int
        The number of time points for generated traces.
    _mu: float
        The base average for traces.
    _sigma: float
        The base standard deviation for traces.
    _noise_amp: float
        The base standard deviation for the noise added to each trace.

    """
    def __init__(self, trace_length, mu, sigma, noise_amp):
        self._trace_length = trace_length
        self._mu = mu
        self._sigma = sigma
        self._noise_amp = noise_amp

    def generate_constant_traces(self, trace_count):
        """Generates `trace_count` constant traces.

        A constant trace is generated by taking a random constant and
        replicating it for the length of the trace. Then for each time point,
        a small bit of noise is added from a `N(0, _noise_amp^2)` random
        variable. The random constant is sampled from a `N(_mu, _sigma^2)`
        random variable.

        Parameters
        ----------
        trace_count: int
            A count of the number of traces to generate.

        Returns
        -------
        list
            A list of numpy arrays representing constant traces. The list has
            `trace_count` traces.

        """
        constants = np.random.normal(self._mu, self._sigma**2, trace_count)
        amp_factor = np.random.normal(1, 1, trace_count)
        traces = [None for _ in range(trace_count)]
        for idx in range(trace_count):
            noise_var = (self._noise_amp ** 2) * abs(amp_factor[idx])
            traces[idx] = constants[idx] + np.random.normal(
                0, noise_var, self._trace_length)
        return traces

    def generate_periodic_traces(self, trace_count, period_avg,
                                 period_std, spike_avg, spike_std):
        """Generates `trace_count` periodic traces.

        A periodic trace is generated by first taking a random constant and
        replicating it for the length of the trace. Then a periodic spike is
        added to the trace. The size of the spike is sampled according to a
        `N(spike_avg, spike_std^2)` random variable and the period of the
        spike is sampled according to a `N(period_avg, period_std^2)` random
        variable. Lastly, for each time point, a small bit of noise is added
        from a `N(0, _amp^2)` random variable.

        Parameters
        ----------
        trace_count: int
            The number of periodic traces being generated.
        period_avg: float
            Represents the average length of the periods. That is, it
            represents the average period length across all generated traces.
        period_std: float
            Represents the standard deviation of period lengths. That is, it
            represents the standard deviation of period lengths across all
            generated traces.
        spike_avg: float
            The average size of the periodic components. That is, it
            represents the average size of the spikes across all periodic
            components added to the generated traces.
        spike_std: float
            The standard deviation of the periodic components. That is, it
            represents the standard deviation of the spikes across all
            periodic components added to the generated traces.

        Returns
        -------
        list
            A list of numpy arrays representing periodic traces. The list has
            `trace_count` traces.

        """
        constants = np.random.normal(self._mu, self._sigma ** 2, trace_count)
        amp_factor = np.random.normal(1, 1, trace_count)
        period_lengths = np.random.normal(
            period_avg, period_std ** 2, trace_count)
        spikes = np.random.normal(spike_avg, spike_std**2, trace_count)
        traces = [None for _ in range(trace_count)]
        for idx in range(trace_count):
            noise_var = (self._noise_amp ** 2) * abs(amp_factor[idx])
            period = round(max(1, abs(period_lengths[idx])))
            offset = random.randint(0, period - 1)
            period_comp = np.array([
                spikes[idx] if (pos - offset) % period == 0 else 0
                for pos in range(self._trace_length)])
            traces[idx] = constants[idx] + period_comp + np.random.normal(
                0, noise_var, self._trace_length)
        return traces
