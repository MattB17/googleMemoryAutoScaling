# Related Work
In [this work](https://www.microsoft.com/en-us/research/wp-content/uploads/2015/08/Harvesting-OSDI16.pdf) the authors investigate placing background batch tasks onto machines running time-sensitive primary tasks
* the goal is to utilize the spare compute power and memory not taken by the primary task to schedule the batch task
* if the presence of the batch task starts to degrade the performance of the primary task then the batch task will be either throttled or killed
* but killing these tasks results in a big performance hit so the goal is to schedule batch tasks in such a way that they are unlikely to be killed
* the approach achieves good results but is a best effort approach, not providing any guarantees on the expected number of tasks killed or an opportunity to tune parameters in order to keep the number of tasks killed below a certain threshold
* to schedule batch tasks, the primary workload are classified into groups after applying the Fast Fourier Transform
* batch tasks are then scheduled onto servers based on the class of the primary workload on that server and a rule based algorithm
* the techniques are implemented in the YARN scheduler, Tez job manager, and HDFS file system

[Starfish](http://cidrdb.org/cidr2011/Papers/CIDR11_Paper36.pdf) investigates the problem of efficient scheduling of Hadoop workflows
* It does so using a profile-predict-optimize approach
* In the profiling phase statistics are collected about a MapReduce job in an attempt to learn performance models for the job
* It uses these statistics to predict job performance based on the job profile and a set of potential Hadoop settings
* these predicted workloads are then handed to a workflow-aware scheduler
* in this work the authors are tuning the Hadoop system to optimize performance
* although our use case is different, the profile-predict-optimize approach is also useful here

[AROMA](https://dl.acm.org/doi/pdf/10.1145/2371536.2371547) is a job provisioning systems for MapReduce jobs
* it determines the number and type of VMs for jobs
* it does so by first clustering jobs based on k-means
* it uses longest common subsequence as the distance metric
* an SVM model is trained for each cluster to predict job performance
* jobs are clustered in an offline phase
* in the online phase, job statistics are collected and longest common subsequence is used
* the SVM model is meant to estimate job completion time whereas this work wants to estimate memory / CPU usage over time

[Autopilot](https://john.e-wilkes.com/papers/2020-EuroSys-Autopilot.pdf) is used at Google to automatically configure compute resources
* it performs horizontal scaling - adjusting the number of machines assigned to a job
* also performs vertical scaling - adjusting CPU and memory limits in existing machines
* this scaling is done in both directions - scaling up when more resources are needed and scaling down when the job is consuming less
* they accept the occasional OOM error, especially in early stages while learning the resources that a task needs
* two types of jobs: serving and batch
* serving jobs generally have SLOs and are the primary driver of capacity while batch jobs generally fill the remaining or temporarily-unused capacity
* it considers jobs separately - there is no cross job learning
* one method computes job limits based on a statistic over the aggregated signal using a moving window (statistic could be max usage, average usage, or a certain percentile)
* also use ML recommenders - a recommender composed of many small interpretable models
  * the recommender periodically chooses the best performing model according to a cost function
* Autopilot solves a different problem - deciding when to autoscale to conserve resources or increase resources to meet demand
* the methods are still useful to predict resource needs of high-priority foreground tasks
* with these predictions, decisions can then be made about how to assign batch jobs to machines to limit the number of tasks that are killed

[This work](https://dl.acm.org/doi/pdf/10.1145/2670979.2670999) focuses on improving the availability SLOs of reclaimed resources
* they want to accurately predict the amount of excess resources which can be reclaimed
* they do this using time-series based forecasting, predicting confidence intervals, and using prediction cycles
* time-series forecasting is done to estimate the available slack (difference between used and available) for a future time period - this is done using ARIMA and Exponential Smoothing
* then they build confidence intervals around the prediction and choose the lowest value in that confidence interval (minimum slack) as the true prediction
* prediction is done in cycles - that is the prediction is made for a window of time and at the end of that time period a new prediction is made
* this overlaps quite closely with our usage in that in this work they are trying to predict usage of foreground jobs
* however these estimates are overly conservative and only focus on minimum availability over a time period
* we instead want to pack batch jobs onto existing servers so it could be the case that a batch job uses more than the minimum slack, but at the same time the foreground job is using less than its maximum and they still use less than the total available resources on the machine

[This work](https://dl-acm-org.myaccess.library.utoronto.ca/doi/pdf/10.1145/2287036.2287050) focuses on predicting disk usage patterns.
* disk space usage, I/O bandwidth, and the age of stored data are measured for distributed filesystems in a cloud environment
* the method starts by analyzing a large quantity of traces
* then usage patterns are aggregated by region and user
* for the users that use high quantities of data and for which their usage patterns are typically hard to predict, these users are required to fill out a quota which is to be incorporated in the forecasts
* next, an ensemble forecasting methodology is used
  * a series of individual forecasts are averaged into one global forecast
  * the forecasting methods used are linear regression, exponential regression, autoregressive integrate moving average (ARIMA), and Bayesian structural time series models
* the method are evaluated on the 3 usage scenarios mentioned above
  * it performs well on short time horizons but requires a buffer for longer horizons
* this bears some similarities to our work, however they are forecasting all jobs
* we are only interested in accurate forecasts of primary jobs and then packing batch jobs onto those existing servers

[PRESS (PRedictive Elastic reSource Scaling)](https://ieeexplore-ieee-org.myaccess.library.utoronto.ca/stamp/stamp.jsp?tp=&arnumber=5691343) focuses on predicting resource patterns and then performing automatic scaling
* goal is to avoid SLO violations and minimize wasted resources
* more weight is attributed to SLO violations as opposed to wasted resources as SLO violations are more costly
* it has a 2 step approach
  * first uses signal processing to identify repeating patterns
  * if no pattern is discovered it uses a statistical state-driven approach to capture near-term resource usage patterns and then uses a discrete-time Markov chain to predict demand in the near future
* for signal processing FFT is first applied to find the longest dominating frequency and then sequences of time windows for the same trace are compared to determine if there are correlations between different time windows that identify a pattern
* the markov chain is meant to predict short-term demand by discretizing usage into equal size buckets and then determining the probability of transitioning from one usage bucket to another for every pair of buckets
* the prediction is then padded with a small buffer to guard against SLO violations
* this has some overlap with our work as it looks to predict the resource needs of jobs
* however, we propose to use a larger collection of methods and to do so only for primary jobs
* batch jobs will then be "packed" onto machines
