import numpy as np
from eosdxanalysis.preprocessing.utils import create_circular_mask

RMIN_BEAM=0
RMAX_BEAM=50


"""
Methods for finding the center of the diffraction pattern.
"""

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
    return tuple(np.mean(points,axis=0))

def find_center(img, mask_center=None, method="max_centroid", rmin=0, rmax=None):
    """
    Find the center of an image in matrix notation

    Output of np.where is a tuple of shape (1,2) with first element
    numpy array of row coordinates, second element numpy array of
    column coordinates. We reshape to (n,2).
    """
    if method == "max_centroid":
        # Create create circular mask for beam region of interest (roi)
        shape = img.shape
        beam_roi = create_circular_mask(shape[0], shape[1],
                center=mask_center, rmin=rmin, rmax=rmax)

        img_roi = np.copy(img)
        img_roi[~beam_roi]=0

        # Find pixels with maximum intensity within beam region of interest (roi)
        # Take tranpose so each rows is coordinates for each point
        max_indices = np.array(np.where(img_roi == np.max(img_roi))).T

        # Find centroid of max intensity
        return find_centroid(max_indices)
    else:
        raise NotImplementedError("Please choose another method.")

def center_of_mass(intensities, visualize=False):
    """
    Given a 2D array of intensities, find the center of mass according to:
    https://en.wikipedia.org/wiki/Center_of_mass#A_system_of_particles
    https://en.wikipedia.org/w/index.php?title=Special:MathWikibase&qid=Q2945123

    Compare to:
    scipy center_of_mass
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.center_of_mass.html

    and numpy average (accepts weights parameter):
    https://numpy.org/doc/stable/reference/generated/numpy.average.html

    and opencv moments:
    https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga556a180f43cab22649c23ada36a8a139
    """
    total_mass = np.sum(intensities)

    R = np.array([0,0])

    for idx in range(intensities.shape[0]):
        # Vectorized code
         A = idx*np.ones((intensities.shape[1],1))
         B = np.arange(intensities.shape[1]).reshape(intensities.shape[1],1)
         r = np.hstack((A, B))
         intensities_row = intensities[idx].reshape(1,intensities.shape[1])
         R = R + np.matmul(intensities_row, r)

    # Normalize by total mass
    R = R.flatten()/total_mass

    # Make x the horizontal and y the vertical index
    xcenter = np.round(R)[1]
    ycenter = np.round(R)[0]

    if visualize:
        # Plot a circle around this point
        fig = plt.figure(dpi=100)
        fig.set_size_inches(4, 4)
        fig.set_facecolor("white")

        plt.imshow(intensities_trunc)

        # Plot circle around center of mass
        # plt.plot(xcenter, ycenter, 'or')
        cir = plt.Circle((xcenter, ycenter), 3, color='r',fill=False)
        plt.gca().add_artist(cir)

        plt.show()

    return R

def radial_mean(intensities,center):
    """
    Given a 2D array of intensities, return the radial mean
    for each intensity value

    Smallest radius is 1 pixel, so 4 pixels is
    the smallest annulus.

    Inputs:
    - intensities: 2D array
    - center: tuple of coordinates for center of 2D array

    """
    ycenter, xcenter = center

    # Create meshgrid
    X,Y = np.meshgrid(np.arange(intensities.shape[1]),np.arange(intensities.shape[0]))
    # Calculate radii
    R = np.sqrt(np.square(X-xcenter)+np.square(Y-ycenter))

    # Calculate the average for each annulus
    radial_mean = np.zeros(intensities.shape)

    # Get unique radii
    radii = np.unique(R).flatten()
    # Get annulus areas for each radius
    areas = np.pi*(np.square(radii+1) - np.square(radii)) 

    for idx in np.arange(radii.shape[0]):
        total_intensity = np.sum(intensities[R == radii[idx]])
        intensity_mean = total_intensity/areas[idx]
        radial_mean[R == radii[idx]] = intensity_mean

    return radial_mean

def radial_histogram(intensities,center):
    """
    Given a 2D array of intensities, return the radial histogram
    for each intensity value along with the corresponding radii

    Smallest radius is 1 pixel, so 4 pixels is
    the smallest annulus.

    Inputs:
    - intensities: 2D array
    - center: tuple of coordinates for center of 2D array

    """
    ycenter, xcenter = center

    # Create meshgrid
    X,Y = np.meshgrid(np.arange(intensities.shape[1]),np.arange(intensities.shape[0]))
    # Calculate radii
    R = np.sqrt(np.square(X-xcenter)+np.square(Y-ycenter))

    # Get unique radii
    radii = np.unique(R).flatten()

    # Store the total intensity for each annulus
    radial_intensities = np.zeros(radii.shape)


    for idx in np.arange(radii.shape[0]):
        annulus_intensity = np.sum(intensities[(R >= radii[idx]) & (R < radii[idx]+1.0)])
        radial_intensities[idx] = annulus_intensity

    return radii, radial_intensities
