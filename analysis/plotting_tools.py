import matplotlib.pyplot as plt

# This function plots a give time series
# FUTURE WORK: add support for multiple series on the same plot
def plot_time_series(time, data, title, xlabel, ylabel, output_file, xlim=None, ylim=None):

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

# This function plots a distribution function (histogram)
def plot_distribution_function(data, bins, title, xlabel, ylabel, output_file):

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

# This function plots a 2D heatmap
def plot_2d_heatmap(data, xlim, ylim, title, xlabel, ylabel, colormap_label, output_file):
    
    plt.figure(figsize=(10, 6))
    plt.imshow(data, aspect='auto', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin='lower', cmap='viridis')
    plt.colorbar(label=colormap_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(output_file)
    plt.close()

