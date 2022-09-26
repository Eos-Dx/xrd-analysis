""""
Visualization helper functions

`heatmap` and `annotated_heatmap` via:
https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
"""
import os
import glob

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_data_dir(input_directory, output_directory, scaling="linear",
        filename_format="*.txt", cmap="hot"):
    """
    Plots raw text data as png files and saves to file.
    """
    input_filenames = glob.glob(os.path.join(input_directory, filename_format))
    input_filenames.sort()

    print("Found " + str(len(input_filenames)) + " files.")

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    basenames = [os.path.basename(fname) for fname in input_filenames]
    barcodes = [os.path.splitext(bname)[0] for bname in basenames]

    for idx in range(len(input_filenames)):

        image = np.loadtxt(input_filenames[idx], dtype=np.uint32)

        if scaling == "linear":
            output_image = image
        if scaling == "dB1":
            # Load image and convert to [dB+1]
            image_dB1 = 20*np.log10(image+1)
            output_image = image_dB1

        # Save image to file
        fname = os.path.join(output_directory, basenames[idx]) + ".png"
        plt.imsave(fname, output_image, cmap=cmap)

def plot_slices(
        input_path, output_path, title="", width=10, N=10, Wn=50, fs=None):
    """
    Plot 1D integrated slices
    """

    # Get dirname of input_path
    fig_suptitle = "{}".format(title)

    # Open figure
    fig = plt.figure(fig_suptitle)
    fig.suptitle(fig_suptitle)

    # Get list of files paths
    filepath_list = glob.glob(os.path.join(input_path, "*"))

    for filepath in filepath_list:
        # Load data
        data = np.loadtxt(filepath, dtype=np.uint32)
        # Rescale data for comparison purposes
        data_rescaled = data/data.sum()*data.size

        if not fs:
            fs = data.shape[0]

        # Filter noise
        # sos = signal.butter(10, 50, 'lp', fs=256, output='sos')
        sos = signal.butter(N=N, Wn=Wn, btype='lp', fs=fs, output='sos')
        data_filtered = signal.sosfilt(sos, data_rescaled)
        # data_filtered = uniform_filter(
        #     uniform_filter(data_rescaled, size=3), size=3)

        # Set up indices for slicing
        center = (data.shape[0]/2-0.5, data.shape[1]/2-0.5)
        row_start = 0; row_end = int(center[0]+0.5);
        col_start = int(center[1]+0.5-width/2);
        col_end = int(center[1]+0.5+width/2);

        # Take 1D slice of filtered data
        data_slice = data_filtered[row_start:row_end, col_start:col_end];
        slice_1d = np.mean(data_slice, axis=1)

        # Plot slice
        plt.plot(slice_1d)

    # Set output image path
    plot_suffix = ".png"
    file_prefix = "{}_slices".format(title) if title else "slices"
    output_filepath = os.path.join(
            output_path, "{}_slices".format(title) + plot_suffix)

    plt.xlabel("Distance from top [pixels]")
    plt.ylabel("Intensity [arbitrary units]")
    plt.ylim([0, 5])
    # plt.show()
    fig.savefig(output_filepath)
    plt.close(fig)
    fig.clear()
