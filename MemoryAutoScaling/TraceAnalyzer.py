"""The `TraceAnalyzer` class is used to calculate statistics and perform
analysis on a trace.

"""
import seaborn as sns
import matplotlib.pyplot as plt


class TraceAnalyzer:
    """Performs analysis on a data trace.

    Parameters
    ----------
    seaborn_style: str
        A string representing the style used for seaborn plots.
    matplotlib_style: str
        A string representing the style used for matplotlib plots.
    tick_interval: int
        An integer representing the interval at which ticks are displayed
        on plots.
    plot_color: str
        A string representing the color for plots
    analysis_title: str
        A string representing the title for the analysis being run.

    Attributes
    ----------
    _tick_interval: int
        The interval at which ticks are displayed on plots.
    _plot_color: str
        The color used for analysis plots.
    _analysis_title: str
        The title of the analysis being run.

    """
    def __init__(self, seaborn_style, matplotlib_style,
                 tick_interval, plot_color, analysis_title):
        sns.set_style(seaborn_style)
        plt.style.use(matplotlib_style)
        self._tick_interval = tick_interval
        self._plot_color = plot_color
        self._analysis_title = analysis_title

    def plot_trace(self, trace):
        """Plots the data trace `trace` over time.

        Parameters
        ----------
        trace: np.array
            A numpy array representing the data trace to be plotted.

        Returns
        -------
        None

        """
        time_points = len(trace)
        tick_labels = ["t{}".format(pos) if pos % self._tick_interval == 0
                       else "" for pos in range(time_points)]
        plt.figure(figsize=(20, 5))
        plt.plot(trace, color=self._plot_color, linewidth=3)
        plt.xticks(range(time_points), tick_labels)
        plt.xlabel("Time")
        plt.title("{} Trace".format(self._analysis_title))
        plt.show()
