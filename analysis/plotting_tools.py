import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

# FUTURE WORK: add support for multiple series on the same plot
def plot_time_series(time: np.array, data: np.array, title: str, xlabel: str, ylabel: str, output_file: str, xlim: Optional[tuple] = None, ylim: Optional[tuple] = None):
    """
    Plots a time series data.

    Args:
        time:                   A numpy array containing time values.
        data:                   A numpy array containing data values corresponding to the time values.
        title:                  The title of the plot.
        xlabel:                 The label for the x-axis.
        ylabel:                 The label for the y-axis.
        output_file:            The filename where the plot will be saved.
        xlim (Optional[tuple]): The limits for the x-axis (min, max). Default is None.
        ylim (Optional[tuple]): The limits for the y-axis (min, max). Default is None.
    """

    plt.figure(figsize=(10, 6))
    plt.plot(time, data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def plot_distribution_function(data: np.array, bins: np.array, title: str, xlabel: str, ylabel: str, output_file: str):
    """
    Plots a distribution function as a histogram.

    Args:
        data:                   A numpy array containing data values.
        bins:                   A numpy array defining the bin edges for the histogram.
        title:                  The title of the plot.
        xlabel:                 The label for the x-axis.   
        ylabel:                 The label for the y-axis.
        output_file:            The filename where the plot will be saved.
    """

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def plot_2d_heatmap(data: np.array, xlim: tuple, ylim: tuple, title: str, xlabel: str, ylabel: str, colormap_label: str, output_file: str):
    """
    Plots a 2d heatmap.

    Args:
        data:                   A 2D numpy array containing the heatmap data.
        xlim:                   A tuple defining the limits for the x-axis (min, max).
        ylim:                   A tuple defining the limits for the y-axis (min, max).
        title:                  The title of the plot.
        xlabel:                 The label for the x-axis.
        ylabel:                 The label for the y-axis.
        colormap_label:         The label for the colorbar.
        output_file:            The filename where the plot will be saved.
    """
    
    plt.figure(figsize=(10, 6))
    plt.imshow(data, aspect='auto', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin='lower', cmap='viridis')
    plt.colorbar(label=colormap_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(output_file)
    plt.close()

