"""A set of utility functions used for plotting.

"""
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt

def get_cdf_values(dist_vals):
    """Computes the cumulative distribution function values for `dist_vals`.

    Parameters
    ----------
    dist_vals: np.array
        A numpy array representing values drawn from a distribution.

    Returns
    -------
    np.array, np.array
        Two numpy arrays representing the x values and y values of the cumulative
        distribution function, respectively.

    """
    dist_vals[np.isnan(dist_vals)] = 0.0
    val_count = len(dist_vals)
    x_vals = np.sort(dist_vals)
    cdf = []
    for i in x_vals:
        cdf.append(len(dist_vals[dist_vals <= i]) / val_count)
    return x_vals, np.array(cdf)


def plot_cumulative_distribution_function(dist_vals, ax, title, color, desc):
    """Plots the cumulative distribution of `dist_vals`.

    Parameters
    ----------
    dist_vals: np.array
        A numpy array representing the values of the distribution for
        which the cumulative distribution function is generated.
    ax: plt.axis
        The axis on which the cumulative distribution is rendered.
    title: str
        A string representing the title for the distribution function.
    color: str
        A string representing the color for the plot.
    desc: str
        A string describing the type of CDF.

    Returns
    -------
    None

    """
    x_vals, cdf = get_cdf_values(dist_vals)
    ax.plot(x_vals, cdf, color=color)
    ax.set_title("{0} - {1}".format(desc, title))


def setup_trace_plot(time_points, tick_interval, title):
    """Performs setup for plots containing trace data.

    Setup involves formatting x axis labels, adding a title and axis labels,
    and displaying the plot.

    Parameters
    ----------
    time_points: int
        An integer representing the number of time points captured in the
        traces that are being plotted.
    tick_interval: int
        An integer representing the interval to display x axis labels. That is,
        for every `tick_interval` time points, a label is printed.
    title: str
        A string representing the plot title.

    """
    tick_labels = ["t{}".format(time_point) if time_point % tick_interval == 0
                   else "" for time_point in range(time_points)]
    plt.xticks(range(time_points), tick_labels)
    plt.xlabel("Time")
    plt.title(title)
    plt.show()


def plot_trace_and_prediction(actual, preds, title):
    """Plots `actual` and its predictions given by `preds`.

    `actual` represents a real trace and `preds` represent a set of
    predictions for the trace. The two are plotted on the same axis
    to see the accuracy of the predictions.

    Parameters
    ----------
    actual: np.array
        A numpy array representing the data trace being plotted.
    preds: np.array
        A numpy array representing the predictions being plotted
    title: str
        A string representing the title for the plot being produced.

    Returns
    -------
    None

    """
    plt.figure(figsize=(20, 8))
    plt.plot(actual, color="blue", linewidth=3, label="Actual")
    plt.plot(preds, color="red", linewidth=3, label="Predicted")
    plt.legend()
    tick_interval = len(actual) // 30
    setup_trace_plot(len(actual), tick_interval, title)


def plot_autocorrelations_for_data(trace, trace_ax, acf_ax,
                                   pacf_ax, lags, color):
    """Plots the autocorrelations for `trace`.

    The raw time series `trace` is plotted as well as its autocorrelation
    and partial autocorrelation based on `lags`.

    Parameters
    ----------
    trace: np.array
        The numpy array representing the data trace.
    trace_ax: plt.axis
        The axis on which the raw time series is plotted.
    acf_ax: plt.axis
        The axis on which the autocorrelations are plotted.
    pacf_ax: plt.axis
        The axis on which the partial autocorrelations are plotted.
    lags: int
        An integer representing the number of lags in the autocorrelation
        plots.
    color: str
        A string representing a color for the plot.

    Returns
    -------
    None

    """
    trace_ax.plot(trace, color=color, linewidth=3)
    smt.graphics.plot_acf(trace, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(trace, lags=lags, ax=pacf_ax)


def plot_actual_vs_predicted_on_axis(actual, predicted, ax, ax_title):
    """Plots `actual` vs `predicted` on `ax`.

    Parameters
    ----------
    actual: np.array
        A numpy array representing actual values.
    predicted: np.array
        A numpy array representing predicted values.
    ax: plt.axis
        A matplotlib axis on which the plot will be rendered.
    ax_title: str
        A string representing the title for the axis.

    Returns
    -------
    None

    """
    ax.plot(actual, color="blue", linewidth=3)
    ax.plot(predicted, color="red", linewidth=3)
    ax.set_title(ax_title)


def plot_train_and_test_predictions_on_axes(y_train, preds_train, y_test,
                                            preds_test, axes, title):
    """Plots actual values vs predicted values on the axes given by `axes`.

    The actual vs predicted values are plotted for both the training and
    testing sets on the different axes of `axes`.

    Parameters
    ----------
    y_train: np.array
        A numpy array containing the actual observations for the training set.
    preds_train: np.array
        A numpy array containing the predictions for the training set.
    y_test: np.array
        A numpy array containing the actual observations for the testing set.
    preds_test: np.array
        A numpy array containing the predictions for the testing set.
    axes: tuple
        A two-element tuple containing the axes on which the plots will be
        rendered.
    title: str
        A string representing the title used for the plots.

    Returns
    -------
    None

    """
    plot_actual_vs_predicted_on_axis(
        y_train, preds_train, axes[0], "{} Training Set".format(title))
    plot_actual_vs_predicted_on_axis(
        y_test, preds_test, axes[1], "{} Testing Set".format(title))
    plt.show()


def plot_multivariate_train_and_test_predictions(
    train_df, train_preds, test_df, test_preds, axes, model_vars, base_title):
    """Plots actual vs predicted values for the variables in `model_vars`.

    The actual vs predicted values are plotted for both the training and
    testing sets for all of the variables in `model_vars` on `axes`. Each
    row consists of two plots for the training and testing sets for a
    particular variable of `model_vars`.

    Parameters
    ----------
    train_df: pd.DataFrame
        A pandas DataFrame containing the target data for the training set.
    train_preds: pd.DataFrame
        A pandas DataFrame containing the predictions for the training set.
    test_df: pd.DataFrame
        A pandas DataFrame containing the target data for the testing set.
    test_preds: pd.DataFrame
        A pandas DataFrame containing the predictions for the testing set.
    axes: tuple
        A tuple containing the aces on which the plots will be rendered. The
        tuple has dimensions `d` by 2 where `d` is the length of `model_vars`.
    model_vars: list
        A list of strings representing the variables being modeled.
    base_title: str
        A string representing the base title used for all plot title.

    Returns
    -------
    None

    """
    for idx in len(model_vars):
        model_var = model_vars[idx]
        plot_actual_vs_predicted_on_axis(
            train_df[model_var].values, train_preds[model_var].values,
            axes[idx, 0],
            "{0} Training Set - {1}".format(base_title, model_var))
        plot_actual_vs_predicted_on_axis(
            test_df[model_var].values, test_preds[model_var].values,
            axes[idx, 1],
            "{0} Testing Set - {1}".format(base_title, model_var))
    plt.show()


def plot_proportions_across_models(model_props, prop_name):
    """Plots the proportions from `model_props` across all models.

    Parameters
    ----------
    model_props: dict
        A dictionary of proportions across models. The keys are strings
        representing the names of models and the associated value is a float
        representing the proportion to be plotted for that model.
    prop_name: str
        A string representing the name of the proportion being plotted.

    Returns
    -------
    None

    """
    print(model_props)
    plt.figure(figsize=(10, 8))
    model_names = model_props.keys()
    model_props = [model_props[model_name] for model_name in model_names]
    plt.scatter(range(len(model_names)), model_props)
    plt.xticks(range(len(model_names)), model_names)
    plt.ylim(-0.05, 1.05)
    plt.ylabel("Prop {}".format(prop_name))
    plt.title("Proportion {} Across Models".format(prop_name))
    plt.show()
