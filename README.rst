Google Memory Auto Scaling
==========================

This package analyzes the Borg traces from Google in May 2019. The original data can be found `here <https://github.com/google/cluster-data>`_. Each trace contains information about maximum memory and CPU usage as well as average memory and CPU usage of a task in periods of 5 minutes.

The purpose of this project is to harvest spare memory and CPU resources from primary jobs running in the Borg cluster. The purpose of this harvesting is to use the spare resources to run secondary batch jobs. In this scenario, priority is given to primary jobs. Thus, the goal is to schedule these batch jobs on the spare resources in such a way that the jobs are rarely throttled or killed because they are stealing resources from the primary jobs.

The code contains functions and scripts for loading raw trace files, performing pre-processing, and modeling the maximum memory usage or maximum CPU usage for each trace.

Installation
------------
To install the package:

* :code:`git clone https://github.com/MattB17/googleMemoryAutoScaling.git`
* :code:`cd MemoryAutoScaling`
* :code:`python -m pip install -r requirements.txt`
* :code:`python -m pip install .`

Scripts
-------
The :code:`Scripts` folder contains python scripts to run models on traces or provide statistical summaries. There are 3 types of scripts:

* :code:`Multivariate` for running multivariate models
  * The multivariate scripts can be run from the command line and take 6 arguments specifying the input directory for trace files, the output directory for the results, the prefix name for traces in the input directory, the minimum length for a trace, the proportion of data in the training set, and an integer representing the number of consecutive periods to aggregate for the modeling
  * For example, the following runs the VARMA models on all traces stored in csv files starting with :code:`task_usage_df` in :code:`<input_dir>`, the traces must have a minimum length of :code:`12`, are aggregated every :code:`3` time periods, with :code:`70%` of data in the training set, and the results are output to :code:`<output_dir>`
    * :code:`python Scripts/Multivariate/buildVARMAModels.py <input_dir> <output_dir> task_usage_df 12 0.7 3`

* :code:`Univariate` for running univariate models

  * The univariate scripts can be run from the command line and take 7 arguments specifying the input directory for trace files, the output directory for the results, the prefix name for traces in the input directory, the minimum length for a trace, the proportion of data in the training set, an integer representing the number of consecutive periods to aggregate for the modeling, and a boolean flag indicating if maximum memory usage is the target variable or maximum CPU usage
  * For example, the following runs the Regression models on all traces stored in csv files starting with :code:`task_usage_df` in :code:`<input_dir>`, the traces must have a minimum length of :code:`12`, are aggregated every :code:`3` time periods, with :code:`70%` of data in the training set, the target variable is maximum memory usage, and the results are output to :code:`<output_dir>`

    * :code:`python Scripts/Univariate/buildRegModels.py <input_dir> <output_dir> task_usage_df 12 0.7 3 True`

* :code:`Statistical` to run statistical summaries

  * The statistical scripts can be run from the command line and take 6 argument specifying the input directory for trace files, the output directory for the results, the prefix name for traces in the input directory, the minimum length for a trace, an integer representing the number of consecutive periods to aggregate for the summary, and a boolean flag indicating if maximum memory usage is the target variable or maximum CPU usage
  * For example, the following runs the statistical analysis on all traces stored in csv files starting with :code:`task_usage_df` in :code:`<input_dir>`, the traces must have a minimum length of :code:`12`, are aggregated every :code:`3` time periods, the target variable is maximum CPU usage, and the results are output to :code:`<output_dir>`

    * :code:`python Scripts/Statistical/getTraceStats.py <input_dir> <output_dir> task_usage_df 12 3 False`

Analysis
--------
The :code:`Notebooks` folder contains jupyter notebooks used to conduct the analysis.

These notebooks assume the scripts in the :code:`Scripts` folder have been run with the appropriate parameters and have been saved in a folder labeled :code:`output_data`.

The results of the statistical analysis on the traces can be found `here <https://docs.google.com/document/d/1K7BBxZMQ5QlbUrKDK4NnTBq--luysnHjai97oCy94HA/edit>`_

The results of the modeling procedures on the traces can be found `here <https://docs.google.com/document/d/16n9JSmnUdko3LTuFWJ0YN_qZpUGBaHSoHmaMigGJYLI/edit#heading=h.fjx4h8ju152c>`_
