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
