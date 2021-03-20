# Models
We fit a series of models to the maximum usage traces

### Linear Regression
We fit a linear regression model to maximum memory usage
* features were previous values for maximum and average memory usage and maximum and average CPU usage


### Moving Average
We fit a Moving Average model to maximum memory usage
* we used windows of size 1, 3, 5, 7, or 10
