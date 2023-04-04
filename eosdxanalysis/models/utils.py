"""
Helper functions for models
"""
import os

import numpy as np
import pandas as pd
from scipy.special import jn_zeros
from scipy.io import savemat
import scipy
import scipy.cluster.hierarchy as sch
from scipy.ndimage import map_coordinates
from collections import OrderedDict

from skimage.transform import rescale
from skimage.transform import warp_polar
from skimage.transform import rotate

from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score

import matplotlib.pyplot as plt

from eosdxanalysis.preprocessing.image_processing import crop_image
from eosdxanalysis.preprocessing.image_processing import pad_image
from eosdxanalysis.preprocessing.utils import create_circular_mask


def gen_jn_zerosmatrix(shape, save_mat=False, save_numpy=False, outdir=""):
    """
    Pre-calculate Bessel zeros
    Can save matlab and/or numpy format
    Used for 2D Polar Discrete Fourier Transform
    """
    nthorder, kzeros = shape
    zeromatrix = np.zeros((nthorder,kzeros))

    for idx in range(nthorder):
        zeromatrix[idx,:] = jn_zeros(idx,kzeros)

    if save_mat:
        # Save to matlab file
        mdic = {"zeromatrix": zeromatrix}
        filename = "zeromatrix.mat"
        full_savepath = os.path.join(outdir, filename)
        savemat(full_savepath, mdic)

    if save_numpy:
        filename = "zeromatrix.npy"
        full_savepath = os.path.join(outdir, filename)
        # Save to numpy file
        np.save(full_savepath, zeromatrix)

    return zeromatrix

def pol2cart(theta, rho):
    """
    Function to convert polar to cartesian coordinates
    Similar to Matlab pol2cart
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def cart2pol(x, y):
    """
    Function to convert cartesian to polar coordinates
    Similar to Matlab cart2pol
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return theta, r

def radial_intensity_1d(image, width=4):
    """
    Returns the 1D radial intensity of positive horizontal strip (averaged).
    For any other strip or quadrant, transpose and/or reverse rows/columns order.
       __
     /    \
    |   ===|
     \ __ /

    """
    # Calculate 1D radial intensity in positive horizontal direction
    center = image.shape[0]/2-0.5, image.shape[1]/2-0.5
    row_start, row_end = int(np.ceil(center[0] - width/2)), int(np.ceil(center[0] + width/2))
    col_start, col_end = int(np.ceil(center[1])), image.shape[1]
    intensity_strip = image[row_start:row_end, col_start:col_end]
    intensity_1d = np.mean(intensity_strip, axis=0) # Average across rows

    return intensity_1d

def angular_intensity_1d(image, radius=None, N=360):
    """
    Get a 1D profile of intensity vs. theta using scipy's
    radial basis function interpolator.

    :param image: image
    :type image: ndarray, two dimensions

    :param radius: radius to take the angular profile at
    :type radius: float

    :param N: output size
    :type N: int

    :returns profile: angular 1D profile
    :rtype: ndarray of length N
    """
    # If radius is not specified, use largest sensible radius
    if radius is None:
        smallest_shape = np.min(image.shape)
        radius = smallest_shape - 0.5
    # Calculate image center
    center = image.shape[0]/2-0.5, image.shape[1]/2-0.5

    # Set up interpolation points in polar coordinates
    start_angle = -np.pi + 2*np.pi/N/2
    end_angle = np.pi - 2*np.pi/N/2
    step = 2*np.pi/N
    angle_array = np.linspace(start_angle, end_angle, num=N, endpoint=True).reshape(-1,1)
    # Set up interpolation points in Cartesian coordinates
    Xcart = radius*np.cos(angle_array)
    Ycart = radius*np.sin(angle_array)
    # Change to array notation
    row_indices = Ycart + center[0]
    col_indices = Xcart + center[1]
    indices = [row_indices, col_indices]

    # Perform interpolation
    angular_profile_1d = map_coordinates(image, indices).flatten()

    return angular_profile_1d

def draw_antialiased_arc(radius, start_angle, angle_spread, output_shape):
    """
    Creates a 2D direc delta function in the shape of an arc.
    The ``angle_spread`` parameter is symmetric about the ``start_angle``.
    See diagram below.
    See ``draw_antialiased_circle`` for reference.

    :param radius: Radius of the arc
    :type radius: float

    :param start_angle: The starting angle of the arc in radians.
        The angle in the positive x direction is `0` radians,
        positive angle is counter-clockwise.
    :type start_angle: float

    :param angle_spread: The angular spread of the arc in radians.
        This is a positive number from `0` to `2pi`.
    :type angle_spread: float

    :param output_shape: Shape of the output
    :type output_shape: ndarray

    :returns arc: The antialiased arc
    :rtype: ndarray


       ___
     / \ / \
    |   V   |
     \ ___ /


    """
    # First calculate the arc as normal, then rotate if needed
    radius = int(radius)

    # Take the modulus of angle_spread with 2*pi
    angle_spread %= 2*np.pi

    # Check if the angle_spread is 0, meaning we want a full circle
    if np.isclose(angle_spread, 0):
        # Use ``draw_antialiased_circle`` function instead
        return draw_antialiased_circle(radius)

    if angle_spread >= np.pi/2:
        raise ValueError("Arc angle spread cannot exceed pi/2.")


    point_array = np.zeros((radius+1, radius+1))
    i = 0
    j = radius
    theta = 0
    last_fade_amount = 0
    fade_amount = 0

    MAX_OPAQUE = 1.0

    # Calculate at most the 1/8th arc
    while i < j and theta < angle_spread/2:
        height = np.sqrt(np.max(radius * radius - i * i, 0))
        fade_amount = MAX_OPAQUE * (np.ceil(height) - height)

        if fade_amount < last_fade_amount:
            # Opaqueness reset so drop down a row.
            j -= 1
        last_fade_amount = fade_amount

        # We're fading out the current j row, and fading in the next one down.
        point_array[i,j] = MAX_OPAQUE - fade_amount
        point_array[i,j-1] = fade_amount

        i += 1
        theta = np.arctan2(i, j)

    image_lower_right = point_array
    # Flip point array vertically (switch order of rows)
    image_upper_right = image_lower_right[::-1, :]
    image_right = np.vstack((image_upper_right, image_lower_right))
    # Image left will be all zeros
    image_left = np.zeros_like(image_right)
    # Join image left and image right
    image = np.hstack((image_left, image_right))

    # Rotate image if specified
    if np.isclose(start_angle, 0):
        return image
    else:
        # Rotation angle is specified radians, convert to degrees
        rotation_degrees = start_angle*180/np.pi
        rotated_image = rotate(image, rotation_degrees, preserve_range=True)
        return rotated_image

def draw_antialiased_circle(outer_radius):
    """
    Adapted via: https://stackoverflow.com/a/37714284

    This will always output a shape 2*(outer_radius+1)

    :param outer_radius: radius of circle
    :type outer_radius: int

    :returns circle: Numpy array of antialiased circle of shape ``2*(outer_radius+1)``
    :rtype: ndarray

    """
    point_array = np.zeros((outer_radius+1, outer_radius+1))

    i = 0
    j = outer_radius
    last_fade_amount = 0
    fade_amount = 0

    MAX_OPAQUE = 1.0

    while i < j:
        height = np.sqrt(np.max(outer_radius * outer_radius - i * i, 0))
        fade_amount = MAX_OPAQUE * (np.ceil(height) - height)

        if fade_amount < last_fade_amount:
            # Opaqueness reset so drop down a row.
            j -= 1
        last_fade_amount = fade_amount

        # We're fading out the current j row, and fading in the next one down.
        point_array[i,j] = MAX_OPAQUE - fade_amount
        point_array[i,j-1] = fade_amount

        i += 1

    # Fully construct the lower-right quadrant by adding the transpose
    quad_lower_right = point_array + point_array.T

    # Flip the lower-right quadrant vertically to get the upper-right quadrant
    quad_upper_right = quad_lower_right[::-1, :]
    # Stack the upper-right and lower-right quadrants to get the right half
    circle_right = np.vstack((quad_upper_right, quad_lower_right))
    # Flip the right half horizontally (row-wise, i.e. reverse order of columns)
    # to get the left half of the circle
    circle_left = circle_right[:,::-1]
    # Finally, stack the left and right halves to form the full circle
    circle = np.hstack((circle_left, circle_right))

    return circle

def l1_metric(A, B):
    """
    Calculates the L1 metric (distance) of two matrices
    based on the L1 norm as defined below.
    """
    if A.size != B.size:
        raise ValueError("Matrix sizes must agree!")
    return 1/A.size*np.sum(abs(A.ravel()-B.ravel()))

def l1_metric_normalized(A, B):
    """
    Calculates the L1 metric (distance) of two matrices
    based on the L1 norm as defined below, with normalization.
    """
    if A.size != B.size:
        raise ValueError("Matrix sizes must agree! Shapes are {}, {}.".format(A.shape, B.shape))
    A_vec = A.ravel()/np.sum(A)
    B_vec = B.ravel()/np.sum(B)
    return np.sum(abs(A_vec-B_vec))

def cluster_corr(corr_array, inplace=False):
    # via: https://wil.yegelwel.com/cluster-correlation-matrix/
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly
    correlated variables are next to eachother

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]

def calculate_min_distance(image1_post, image2_post_unmasked, mask, scale=1.0,
                        tol=1e-2, max_iters=1e1, iterations=0):
    iterations = iterations+1

    h, w = image1_post.shape
    distance = l1_metric_normalized(image1_post, image2_post_unmasked)

    if iterations > max_iters:
        return distance

    # These images may be have odd shape
    image2_smallscale_unmasked = rescale(image2_post_unmasked, scale*0.9, anti_aliasing=False)
    image2_largescale_unmasked = rescale(image2_post_unmasked, scale*1.1, anti_aliasing=False)

    # First centerize the smaller image so that it is even
    image2_smallscale_center = (image2_smallscale_unmasked.shape[0]/2,
                            image2_smallscale_unmasked.shape[1]/2,)
    image2_smallscale_centerized, _ = centerize(image2_smallscale_unmasked,
                            image2_smallscale_center)
    # Crop centerized image to 256x256
    image2_smallscale_unmasked = crop_image(image2_smallscale_centerized, h, w)
    # Crop the larger image to 256x256
    image2_largescale_unmasked = crop_image(image2_largescale_unmasked, h, w)

    # Mask images
    image2_smallscale = image2_smallscale_unmasked.copy()
    image2_largescale = image2_largescale_unmasked.copy()
    image2_smallscale[~mask] = 0
    image2_largescale[~mask] = 0

    distance_smallscale = l1_metric_normalized(image1_post, image2_smallscale)
    distance_largescale = l1_metric_normalized(image1_post, image2_largescale)

    if np.abs(distance_smallscale - distance) < tol:
        return distance_smallscale
    elif np.abs(distance_largescale - distance) < tol:
        return distance_largescale
    elif distance_smallscale < distance:
        # Free up memory
        del image2_smallscale_centerized
        del image2_largscale_unmasked
        del image2_smallscale
        del image2_largescale
        # Set new scale
        new_scale = scale*0.9
        return calculate_min_distance(
                image1_post, image2_smallscale_unmasked, mask,
                scale=new_scale, tol=tol, iterations=iterations)
    else:
        # Free up memory
        del image2_smallscale_centerized
        del image2_smallscale_unmasked
        del image2_smallscale
        del image2_largescale
        # Set new scale
        new_scale = scale*1.1
        return calculate_min_distance(
                image1_post, image2_largescale_unmasked, mask,
                scale=new_scale, tol=tol, iterations=iterations)

def metrics_report(
        TP=0, FP=0, TN=0, FN=0, print_csv=True, print_table=False):
    """
    Generate a metrics report from blind test results

    Parameters
    ----------

    TP : int
        True positives

    FP : int
        False positives

    TN : int
        True negatives

    FN : int
        False negatives

    """

    # Calculate accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # Calculate false positive rate
    FPR = FP / (FP + TN)
    # Calculate false negative rate
    FNR = FN / (FN + TP)
    # Calculate precision
    precision = TP / (TP + FP)
    # Calculate recall (sensitivity)
    sensitivity = TP / (TP + FN)
    # Specificity
    specificity = TN / (TN + FP)
    # Calculate F1 score
    F1 = 2 * precision * sensitivity / (precision + sensitivity)

    metrics_dict = OrderedDict({
            "TN": [TN, ".0f"],
            "FP": [FP, ".0f"],
            "FN": [FN, ".0f"],
            "TP": [TP, ".0f"],
            "FPR": [FPR, ".2f"],
            "FNR": [FNR, ".2f"],
            "Accuracy": [accuracy, ".2f"],
            "Precision": [precision, ".2f"],
            "Sensitivity": [sensitivity, ".2f"],
            "Specificity": [specificity, ".2f"],
            "F1": [F1, ".2f"],
            })

    if print_table:
        data = np.array(tuple(metrics_dict.values()))
        df = pd.DataFrame(data,
                index=metrics_dict.keys())

        print(df.to_markdown(index=False,floatfmt=format_list))

    if print_csv:
        metric_names = tuple(metrics_dict.keys())
        metric_data = np.array(tuple(metrics_dict.values()), dtype=object)
        for idx in range(len(metric_names)):
            metric_name = metric_names[idx]
            metric_value, format_string = metric_data[idx]
            print(f"{metric_name},{metric_value:{format_string}}")

def gen_meshgrid(shape):
    """
    Generate a meshgrid
    """
    # Generate a meshgrid the same size as the image
    x_end = shape[1]/2 - 0.5
    x_start = -x_end
    y_end = x_start
    y_start = x_end
    YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
    TT, RR = cart2pol(XX, YY)

    return RR, TT

def scale_features(df, scale_by, feature_list):
    """
    Scale all features by ``scale_by``
    Returns a copy of a dataframe
    Drops ``scale_by`` column
    """
    df_scaled_features = df[feature_list].div(
            df[scale_by], axis="rows")
    if scale_by in feature_list:
        # Drop ``scale_by`` column
        df_scaled_features = df_scaled_features.drop(columns=[scale_by])
    return df_scaled_features

def add_patient_data(df, patient_db_filepath, index_col="Barcode"):
    # Use patient database to get patient diagnosis
    db = pd.read_csv(patient_db_filepath, index_col=index_col)
    extraction = df.index.str.extractall(
            "CR_([A-Z]{1}).*?([0-9]+)")
    extraction_series = extraction[0] + extraction[1].str.zfill(5)
    extraction_list = extraction_series.tolist()

    assert(len(extraction_list) == df.shape[0])
    df_ext = df.copy()
    df_ext[index_col] = extraction_list

    df_ext = pd.merge(
            df_ext, db, left_on=index_col, right_index=True)

    return df_ext

def plot_roc_curve(
        normal_cluster_list=None, cancer_cluster_list=None,
        y_true_patients=None, y_score_patients=None, threshold_range=None,
        sensitivity_array=None, specificity_array=None, subtitle=None):
    # Calculate ROC AUC
    auc = roc_auc_score(y_true_patients, y_score_patients)

    # Manually create ROC and precision-recall curves
    tpr = sensitivity_array
    fpr = 1 - specificity_array
    x_offset = 0
    y_offset = 0.002

    # ROC Curve
    title = "ROC Curve - {}".format(subtitle)
    fig = plt.figure(title, figsize=(12,12))

    if normal_cluster_list in (None, ""):
        plt.step(fpr, tpr, where="post", label="AUC = {:0.2f}".format(auc))
    if cancer_cluster_list in (None, ""):
        plt.step(fpr, tpr, where="pre", label="AUC = {:0.2f}".format(auc))

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title(title)
    plt.legend(loc="lower right")

    # Annotate by threshold
    for x, y, s in zip(fpr, tpr, threshold_range):
        plt.text(x+x_offset, y+y_offset, np.round(s,1))

    plt.show()

def plot_precision_recall_curve(
        normal_cluster_list=None, cancer_cluster_list=None,
        threshold_range=None, recall_array=None, precision_array=None,
        subtitle=None):
    # Precision-Recall Curve
    title = "Precision-Recall Curve - {}".format(subtitle)
    fig = plt.figure(title, figsize=(12,12))

    x_offset = 0
    y_offset = 0.002

    if normal_cluster_list in (None, ""):
        plt.step(recall_array, precision_array, where="pre")
    if cancer_cluster_list in (None, ""):
        plt.step(recall_array, precision_array, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title(title)

    # Annotate by threshold
    for x, y, s in zip(recall_array, precision_array, threshold_range):
        plt.text(x+x_offset, y+y_offset, np.round(s,1))

    plt.show()

def plot_patient_score_scatterplot(
        y_true_patients, y_score_patients,
        y_test_score_patients=None):
    # Plot scatter plot
    # Calcualte mean of healthy and cancer patient scores
    y_score_healthy_patients = y_score_patients[~(y_true_patients.astype(bool))]
    y_score_healthy_patients_mean = np.mean(y_score_healthy_patients)
    y_score_cancer_patients = y_score_patients[y_true_patients.astype(bool)]
    y_score_cancer_patients_mean = np.mean(y_score_cancer_patients)

    healthy_label = "Healthy Mean: {:.1f}".format(
            y_score_healthy_patients_mean)
    cancer_label = "Cancer Mean: {:.1f}".format(
            y_score_cancer_patients_mean)

    plt.scatter(y_score_healthy_patients,
            np.zeros(y_score_healthy_patients.shape),
            c="blue", label=healthy_label)
    plt.scatter(y_score_cancer_patients,
            np.ones(y_score_cancer_patients.shape),
            c="red", label=cancer_label)

    if y_test_score_patients is not None:
        y_test_score_patients_mean = np.mean(y_test_score_patients)
        blind_label = "Blind Mean: {:.1f}".format(
                y_test_score_patients_mean)
        plt.scatter(y_test_score_patients,
                -np.ones(y_test_score_patients.shape),
                c="green", label=blind_label)

    plot_title = "Scatterplot of Patient Scores"

    plt.title(plot_title)
    plt.xlabel("Patient Score")
    plt.legend(loc="upper right")
    plt.ylim([-5, 5])
    plt.show()

def plot_patient_score_histogram(y_score_patients):
    # Plot training patient scores histogram
    y_score_patients_mean = np.mean(y_score_patients)
    label = "Mean: {:.1f}".format(y_score_patients_mean)
    plt.hist(y_score_patients, label=label)

    plot_title = "Histogram of Patient Scores"

    plt.title(plot_title)
    plt.ylabel("Frequency Count")
    plt.xlabel("Patient Score")
    plt.legend(loc="upper right")
    plt.show()



def  specificity_score(y_true, y_pred):
    """
    Returns specificity score
    """
    return recall_score(y_true, y_pred, pos_label=0)


#def l1_metric_optimized(image1, image2, params, plan=None):
#    """
#    Function which computes the L1 distance between two images
#    that may include a sample-to-detector distance shift.
#    The algorithm performs a binary search to minimize the distance
#    between one image and resized version of the other image.
#    """
#    # Set the tolerance for convergence criterion
#    TOL=1e-6
#
#    if plan is None:
#        plan = [
#                "local_thresh_quad_fold",
#                ]
#        output_style = [
#                "local_thresh_quad_folded",
#                ]
#
#    params1 = params.copy()
#    params2 = params.copy()
#    del params2["crop_style"]
#
#    # Preprocess both images according to parameters, not cropping image2
#    image1_preprocessor = PreprocessDataArray(image1, params=params1)
#    image2_preprocessor = PreprocessDataArray(image2, params=params2)
#
#    # Preprocess images
#    image1_preprocessor.preprocess(plan, mask_style="both")
#    image2_preprocessor.preprocess(plan, mask_style=None)
#    image1_post = image1_preprocessor.cache.get(output_style[0])
#    image2_post_unmasked = image2_preprocessor.cache.get(output_style[0])
#
#    # Crop image2
#    image2_preprocessor.preprocess(plan, mask_style="both")
#    image2_post = image2_preprocessor.cache.get(output_style[0])
#
#    # Create mask
#    h, w = image1.shape
#    rmin = params.get("rmin")
#    rmax = params.get("rmax")
#    mask = create_circular_mask(h, w, rmin=rmin, rmax=rmax)
#
#    distance = calculate_min_distance(
#                    image1_post, image2_post_unmasked, mask, tol=TOL)
#
#    return distance
