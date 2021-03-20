# Trace Statistics
Statistics were gathered for high-priority traces with at least 10 observations.
* this resulted in 2,701 traces
* for these traces we are interested in predicting the maximum memory usage for future time periods based on past usage statistics

### Stationarity
One statistic examined was the [Augmented Dickey-Fuller test for stationarity](https://www-jstor-org.myaccess.library.utoronto.ca/stable/2286348?sid=primo&origin=crossref&seq=1#metadata_info_tab_contents)
* this test determines whether a time series exhibits stationarity - that is, it does not have an underlying trend.

The stationarity test was applied to each time series, as well as the time series that resulted from one and two levels of differencing
* one level differencing subtracts the past value of the time series from the current value to get the net change since the last period
* two level differencing is differencing applied to the time series obtained from one level differencing
* 37% of traces were stationary without any differencing
* 51% were stationary after 1 level of differencing
* 7% were stationary after 2 levels of differencing
* and the remainder were still not stationary after two levels of differencing

### Correlation
As well as maximum memory usage, average memory usage, average CPU usage, and maximum CPU usage are measured for each trace.

For each trace, the correlation of each of these other traces and maximum memory usage was computed.
* for 99.7% of traces, the correlation between maximum memory usage and average memory usage was above 0.7 or below -0.7
* for 33.8% of traces, the correlation between maximum memory usage and maximum CPU usage was above 0.7 or below -0.7
* for 29.2% of traces, the correlation between maximum memory usage and average CPU usage was above 0.7 or below -0.7

### Causality
The Granger test for causality was applied to see if any of average memory usage, average CPU usage, or maximum CPU usage was causally related to maximum memory usage
* as with correlation, it was found that for most traces, the hypothesis that past values of maximum memory usage and average memory usage were causally related to current maximum memory usage were much stronger than those for maximum and average CPU usage
* this shows up in the models.
