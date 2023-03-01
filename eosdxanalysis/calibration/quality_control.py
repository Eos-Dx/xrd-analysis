"""
The file is to provide automated quality control tool for technicians to check the initial beam quality
It checks the following:
  1. The beam detector alignment
  2. The beam aperture alignment
"""

import numpy as np
import matplotlib.pyplot as plt

def find_centroid(points):
    """
    Given an array of shape (n,2), with elemenets aij,
    [[a00,a01],
     [a10,a11],
        ...],
    calculate the centroid in row, column notation.

    Returns a tuple result with row and column centroid
    """
    try:
        shape = points.shape
        dim = shape[1]
        if dim != 2:
            raise ValueError("Input must be array of shape (n,2)!")
    except AttributeError as error:
        print(error)
        raise AttributeError("Input must be array of shape (n,2)!")
    except IndexError as error:
        print(error)
        raise ValueError("Input must be array of shape (n,2)!")

    # Return centroid
    return tuple(np.mean(points, axis=0))


def find_weighted_centroid(input_image):
    # The weighted sum of all data coordinates divided by the total weight
    weighted_sum = np.array([0, 0])
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            weighted_sum = weighted_sum + np.array([i, j]) * input_image[i][j]

    weighted_centroid = weighted_sum / input_image.sum()

    return weighted_centroid


def check_beam_detector_alignment(input_filepath, error_tol=np.array([5, 5])):
    """
    Prints the beam position
    Comments on whether the beam position is within the tolerance
    """

    # load the image
    image = np.loadtxt(input_filepath)

    # calculating the weighted centroid
    weighted_centroid = find_weighted_centroid(image)
    print("Beam position: {:.1f}, {:.1f}".format(
        weighted_centroid[1], weighted_centroid[0] - weighted_centroid.shape[0]))

    # calculate error
    detector_center = np.array([127.5, 127.5])
    error = weighted_centroid - detector_center
    print("Beam position error: {:.1f}, {:.1f}".format(error[1], -error[0]))

    # check tolerance
    # error_tol = np.array([5, 5])
    if (np.abs(error) > error_tol).any():
        print("Beam position is out of bounds!")
    else:
        print("Beam position is within tolerance.")

    return weighted_centroid


def check_beam_aperture_alignment(input_filepath, error_tol=np.array([2, 2])):
    """
    Prints the beam aperture center position
    Comments whether the beam aperture position is within the tolerance
    """

    # load the image
    image = np.loadtxt(input_filepath)

    # calculating the weighted centroid
    weighted_centroid = find_weighted_centroid(image)

    # calculating the beam aperture center position
    image_max_coordinates = np.array(np.where(image == image.max())).T
    max_centroid = np.array(find_centroid(image_max_coordinates))
    print("Max centroid position: {:.1f}, {:.1f}".format(
        max_centroid[1], max_centroid[0] - max_centroid.shape[0]))

    error = weighted_centroid - max_centroid
    print("Beam max centroid position error: {:.1f}, {:.1f}".format(error[1], -error[0]))

    # error_tol = np.array([2, 2])
    if (np.abs(error) > error_tol).any():
        print("Beam-aperture alignment is out of bounds!")
    else:
        print("Beam-aperture alignment is within tolerance.")

    return max_centroid

def plot_centers(input_filepath):
    """
    Plots the given file as well as the Beam position centre and the Max Centroid position
    """
    # Load the image
    image = np.loadtxt(input_filepath)

    # Plots the data given
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="hot")

    # Plot the real centre
    real_centre = np.array([127.5, 127.5])
    ax.scatter(real_centre[0], real_centre[1], label="detector centre", marker="o")

    # Plot the Beam position centre
    weighted_centroid = check_beam_detector_alignment(input_filepath)
    ax.scatter(weighted_centroid[1], weighted_centroid[0], label="weighted centroid", marker="+")

    # Plot the max centroid position
    max_centroid = check_beam_aperture_alignment(input_filepath)
    ax.scatter(max_centroid[1], max_centroid[0], label="max centroid", marker="*")

    # legend
    plt.legend()

    plt.show()
    return image


if __name__ == '__main__':
    """
    Check if the machine has proper beam-detector alignment
    """
    measurement_filepath = input(
            "Enter the full path to the beam-only measurement file:\n")

    # Remove extra spaces or quote characters at the ends
    measurement_filepath = measurement_filepath.strip(" ").strip("\'")

    plot_centers(measurement_filepath)
