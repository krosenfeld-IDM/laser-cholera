import click
import matplotlib.pyplot as plt
import numpy as np
from laser_core.migration import distance
from matplotlib.backends.backend_pdf import PdfPages


def calc_distances(latitudes: np.ndarray, longitudes: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Calculate the pairwise distances between points given their latitudes and longitudes.

    Parameters:

        latitudes (np.ndarray): A 1-dimensional array of latitudes.
        longitudes (np.ndarray): A 1-dimensional array of longitudes with the same shape as latitudes.
        verbose (bool, optional): If True, prints the upper left corner of the distance matrix. Default is False.

    Returns:

        np.ndarray: A 2-dimensional array where the element at [i, j] represents the distance between the i-th and j-th points.

    Raises:

        AssertionError: If latitudes is not 1-dimensional or if latitudes and longitudes do not have the same shape.
    """

    assert latitudes.ndim == 1, "Latitude array must be one-dimensional"
    assert longitudes.shape == latitudes.shape, "Latitude and longitude arrays must have the same shape"
    npatches = len(latitudes)
    distances = np.zeros((npatches, npatches), dtype=np.float32)
    for i, (lat, long) in enumerate(zip(latitudes, longitudes)):
        distances[i, :] = distance(lat, long, latitudes, longitudes)

    if verbose:
        click.echo(f"Upper left corner of distance matrix:\n{distances[0:4, 0:4]}")

    return distances


def viz_twodee(data, title, x_label, y_label, pdf: bool = False):
    # Set up the figure with a specific aspect ratio
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create the heatmap with 1 pixel per value
    cax = ax.imshow(data, cmap="viridis", interpolation="none", aspect="auto", origin="lower")

    # Adjust color bar size and label
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Value")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if pdf:
        with PdfPages(title.replace(" ", "_") + ".pdf") as pdf:
            pdf.savefig()
            plt.close()
    else:
        plt.show()

    return
