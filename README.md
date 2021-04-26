# Google Memory Auto Scaling

This package analyzes the Borg traces from Google in May 2019. The original data can be found [here](https://github.com/google/cluster-data). Each trace contains information about maximum memory and CPU usage as well as average memory and CPU usage of a task in periods of 5 minutes.

The purpose of this project is to harvest spare memory and CPU resources from primary jobs running in the Borg cluster. The purpose of this harvesting is to use the spare resources to run secondary batch jobs. In this scenario, priority is given to primary jobs. Thus, the goal is to schedule these batch jobs on the spare resources in such a way that the jobs are rarely throttled or killed because they are stealing resources from the primary jobs.

The code contains functions and scripts for loading raw trace files, performing pre-processing, and modeling the maximum memory usage or maximum CPU usage for each trace.

### Installation
To install the package:
* `git clone https://github.com/MattB17/googleMemoryAutoScaling.git`
* `cd MemoryAutoScaling`
* `python -m pip install -r requirements.txt`
* `python -m pip install .`
