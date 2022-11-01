"""
Performs preprocessing pipeline
"""

import os
import glob
import argparse
import json
from datetime import datetime
import subprocess

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors

from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.filters import threshold_local
from skimage.transform import EuclideanTransform
from skimage.transform import warp
from skimage.transform import rotate

from eosdxanalysis.preprocessing.center_finding import find_center
from eosdxanalysis.preprocessing.center_finding import find_centroid
from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.preprocessing.utils import get_angle
from eosdxanalysis.preprocessing.utils import find_maxima
from eosdxanalysis.preprocessing.denoising import find_hot_spots
from eosdxanalysis.preprocessing.denoising import filter_hot_spots
from eosdxanalysis.preprocessing.image_processing import crop_image
from eosdxanalysis.preprocessing.image_processing import quadrant_fold
from eosdxanalysis.preprocessing.beam_utils import beam_extent

from eosdxanalysis.simulations.utils import feature_pixel_location



ABBREVIATIONS = {
        # output style: output style abbreviation
        "original": "O",
        "centered": "C",
        "centered_rotated": "CR",
        "centered_rotated_quad_folded": "CRQF",
        "local_thresh_centered_rotated": "LTCR",
        "local_thresh_centered_rotated_quad_folded": "LTCRQF",
        }

OUTPUT_MAP = {
        # Maps output style to input plan
        "original":"original",
        "centered":"centerize",
        "centered_rotated":"centerize_rotate",
        "centered_rotated_quad_folded":"centerize_rotate_quad_fold",
        "local_thresh_centered_rotated":"local_thresh_centerize_rotate",
        "local_thresh_centered_rotated_quad_folded":"local_thresh_centerize_rotate_quad_fold",
        }

INVERSE_OUTPUT_MAP = {
        # Maps preprocessing plan to output style
        "original":"original",
        "centerize":"centered",
        "centerize_rotate":"centered_rotated",
        "centerize_rotate_quad_fold":"centered_rotated_quad_folded",
        "local_thresh_centerize_rotate":"local_thresh_centered_rotated",
        "local_thresh_centerize_rotate_quad_fold":"local_thresh_centered_rotated_quad_folded",
        }


class PreprocessData(object):
    """
    Class to run processing pipeline on data set.

    Initialize the ``PreprocessData`` class with input, output (optional),
    and parameters.

    Parameters
    ----------

    input_path : str

        Full path to location of input files. If ``data_dir`` is specified,
        ``input_path`` is the parent directory of ``data_dir``.

    filename : str

        Name of single file to preprocess.

    data_dir : str (optional)

        Name of subdirectory in ``input_path`` containing data files.

    output_path : str (optional)

        Full path to store output files.

    params : dict

    Notes
    -----

    Preprocessing parameters are defined as follows:

    * ``h``: image height
    * ``w``: image width
    * ``beam_rmax``: defines circular region of interest of the beam
    * ``rmin``: final image masking inner masking
    * ``rmax``: final image masking outer masking
    * ``eyes_rmin``: 9 A region of interest, inner radius of annulus
    * ``eyes_rmax``: 9 A region of interest, outer radius of annulus
    * ``eyes_blob_rmax``: defines a circle region of interest centered at the eye peak
    * ``eyes_percentile``: defines the threshold for generating a binary blob for noise-robust 9 A peak finding
    * ``local_thresh_block_size``: no longer used (future: specify filter type and size)
    * ``crop_style``: choice of ``"both"``, ``"beam"``, or ``"outside"``. ``"beam"`` sets the inner circle of radius ``rmin`` to zero. ``"outside"`` sets values outside ``rmax`` to zero. ``"both"`` does both.
    * ``plans``: a list of strings denoting the preprocessing plan(s) to perform. Choice of ``"original"``, ``"centerize"``, ``"centerize_rotate"``, and ``"centerize_rotate_quad_fold"``. (Note: JSON syntax does not allow for a spare comma at the end of a list, whereas Python does.)

    """

    def __init__(self, input_path=None, filename=None, data_dir=None,
            output_path=None, params={}):
        # Store input and output directories
        self.input_path = input_path
        self.data_dir = data_dir
        self.output_path = output_path
        # Set filename
        self.filename = filename

        file_path_list = []

        # Handle various cases
        # We are given a filename
        if filename and input_path:
            # Collect full file fullpath into a single-element list
            file_path = os.path.join(input_path, filename)
            file_path_list = [file_path]
        # We are given an input path and a data subdirectory
        elif input_path and data_dir:
            # Get sorted list of filenames
            file_path_list = glob.glob(
                    os.path.join(input_path, data_dir, "*.txt"))
            file_path_list.sort()
        # Else we are provided with an input directory only
        elif input_path:
            # Get sorted list of filenames
            file_path_list = glob.glob(os.path.join(input_path,"*.txt"))
            file_path_list.sort()
        else:
            pass

        # Store filenames (these are full paths)
        self.file_path_list = file_path_list

        # Store parameters
        self.params = params

        return super().__init__()

    def preprocess(self, plans=["centerize"],
                mask_style="both", uniform_filter_size=0, scaling="linear"):
        """
        Run all preprocessing steps

        Plan options include:
        - centerize
        - centerize_rotate
        - quad_fold
        - local_thresh_centerize_rotate
        - local_thresh_quad_fold

        Mask options include:
        - beam
        - corners
        - both
        """
        # Get and set parameters
        params = self.params
        h = params.get("h")
        w = params.get("w")
        beam_rmax = params.get("beam_rmax")
        rmin = params.get("rmin")
        rmax = params.get("rmax")
        eyes_rmin = params.get("eyes_rmin")
        eyes_rmax = params.get("eyes_rmax")
        eyes_blob_rmax = params.get("eyes_blob_rmax")
        eyes_percentile = params.get("eyes_percentile")
        local_thresh_block_size = params.get("local_thresh_block_size")
        beam_detection = params.get("beam_detection")
        cmap = params.get("cmap", "hot")
        hot_spot_threshold = params.get("hot_spot_threshold", None)

        # Get plans from from parameters or keyword argument
        plans = params.get("plans", plans)

        # Set mask style from params if crop_style is set
        mask_style = params.get("crop_style", mask_style)

        # Get filename info
        file_path_list = self.file_path_list

        # Set timestamp
        timestr = "%Y%m%dT%H%M%S.%f"
        timestamp = datetime.utcnow().strftime(timestr)
        self.timestamp = timestamp

        # Set paths
        input_path = self.input_path
        output_path = self.output_path
        data_dir = self.data_dir

        # Store output directory info
        if not self.output_path:
            # Create output directory if it does not exist
            if data_dir:
                output_dir = "preprocessed_{}_{}".format(
                                        data_dir, timestamp)
            else:
                output_dir = "preprocessed_{}".format(timestamp)
            output_path = os.path.join(input_path, output_dir)
            os.makedirs(output_path, exist_ok=True)
        else:
            output_path = self.output_path
            os.makedirs(output_path, exist_ok=True)

        print("Saving to", output_path, "...")

        # Write params to file
        with open(os.path.join(output_path,"params.txt"),"w") as paramsfile:
            paramsfile.write(json.dumps(params,indent=4))

        # Loop over plans
        for plan in plans:

            output_style = INVERSE_OUTPUT_MAP.get(plan)
            output_style_abbreviation = ABBREVIATIONS.get(output_style)

            # Create output directory for plan and output format
            plan_output_path = os.path.join(output_path, output_style)
            os.makedirs(plan_output_path, exist_ok=True)

            # Create output directory for images
            plan_output_images_path = os.path.join(output_path, output_style + "_images")
            os.makedirs(plan_output_images_path, exist_ok=True)

            # Loop over files
            for file_path in file_path_list:
                # Load file
                plan_image = np.loadtxt(file_path)

                # Calculate array center
                array_center = np.array(plan_image.shape)/2-0.5
                self.array_center = array_center
                center = None

                filename = os.path.basename(file_path)

                if hot_spot_threshold:
                    # Use a beam-masked image to filter hot spots
                    center = find_center(plan_image, method="max_centroid", rmax=beam_rmax)
                    mask = create_circular_mask(h, w, center=center, rmin=beam_rmax)
                    masked_image = plan_image.copy()
                    masked_image[~mask] = 0
                    hot_spot_coords_array = find_hot_spots(masked_image, hot_spot_threshold)
                    # Filter hot spots on original image, not masked image
                    plan_image = filter_hot_spots(
                            plan_image, threshold=hot_spot_threshold,
                            hot_spot_coords_array=hot_spot_coords_array)

                # Set the output based on output specifications
                if plan == "original":
                    output = plan_image

                elif plan == "centerize":
                    # Centerize and rotate
                    centered_image, center = self.centerize(plan_image)
                    # Set output
                    output = centered_image

                elif plan == "centerize_rotate":
                    # Centerize and rotate
                    centered_rotated_image, center, angle = self.centerize_and_rotate(plan_image)
                    # Set output
                    output = centered_rotated_image

                elif plan == "centerize_rotate_quad_fold":
                    # Centerize and rotate
                    centered_rotated_image, center, angle = self.centerize_and_rotate(plan_image)
                    # Quad fold
                    centered_rotated_quad_folded_image = quadrant_fold(centered_rotated_image)
                    # Set output
                    output = centered_rotated_quad_folded_image

                elif plan == "local_thresh_centerize_rotate":
                    # Take local threshold
                    local_thresh_image = threshold_local(plan_image, local_thresh_block_size)
                    # Centerize and rotate
                    local_thresh_centered_rotated_image, center, angle = self.centerize_and_rotate(local_thresh_image)
                    # Set output
                    output = local_thresh_centered_rotated_image

                elif plan == "local_thresh_centerize_rotate_quad_fold":
                    # Take local threshold
                    local_thresh_image = threshold_local(plan_image, local_thresh_block_size)
                    # Centerize and rotate
                    local_thresh_centered_rotated_image, center, angle = self.centerize_and_rotate(local_thresh_image)
                    # Quad fold
                    local_thresh_centered_rotated_quad_folded_image = quadrant_fold(local_thresh_centered_rotated_image)
                    # Set output
                    output = local_thresh_centered_rotated_quad_folded_image

                # Uniform filter
                if uniform_filter_size > 1:
                    output = ndimage.uniform_filter(output, size=uniform_filter_size)

                # Mask
                if mask_style:
                    if not center:
                        center = find_center(plan_image, method="max_centroid", rmax=beam_rmax)
                    output = self.mask(output, center=center, style=mask_style)

                # Save the file
                output_filename = "{}_{}".format(output_style_abbreviation,
                                            filename)
                output_file_path = os.path.join(plan_output_path,
                                            output_filename)
                np.savetxt(output_file_path,
                                np.round(output).astype(np.uint32), fmt='%i')

                # Save the image
                output_image_filename = output_filename + ".png"
                output_image_path = os.path.join(plan_output_images_path,
                        output_image_filename)
                if scaling == "dB1":
                    output = 20*np.log10(output+1)
                plt.imsave(output_image_path, output, cmap=cmap)


    def find_eye_rotation_angle(self, image, center):
        """
        Find the rotation angle of the original XRD pattern using the following steps:
        1. Find centroid maximum intensities in 9A "eye" feature region
        2. Convert 9A features to binary blobs and find centroid
        3. Calculate rotation angle from image center to blob centroid

        We work on the original image here to find the rotation angle,
        not on the centered and rotated image.

        Inputs:
        - image
        - center of image
        """
        params = self.params
        h = params.get("h")
        w = params.get("w")
        beam_rmax = params.get("beam_rmax")
        rmin = params.get("rmin")
        rmax = params.get("rmax")
        eyes_rmin = params.get("eyes_rmin")
        eyes_rmax = params.get("eyes_rmax")
        eyes_blob_rmax = params.get("eyes_blob_rmax")
        eyes_percentile = params.get("eyes_percentile")
        local_thresh_block_size = params.get("local_thresh_block_size")
        gauss_filter_size = params.get("gauss_filter_size", 3)

        # 1. Find the 9A arc maxima features

        # Perform Gaussian filtering on original
        filtered_image = gaussian_filter(image, gauss_filter_size)

        # Apply circular mask as a digital beamstop to center
        beam_mask = create_circular_mask(h,w,center=center,rmin=rmin,rmax=rmax)
        masked_image = np.copy(image)
        masked_image[~beam_mask] = 0

        # Create mask for eye region
        eye_mask = create_circular_mask(h,w,center=center,rmin=eyes_rmin,rmax=eyes_rmax)

        # Use filtered_image results
        eye_roi = np.copy(filtered_image)
        eye_roi[~eye_mask] = 0

        # Find the peaks in the 9A region using eye roi
        peak_location_radius_9A_theory = feature_pixel_location(9e-10)
        peaks = peak_local_max(eye_roi,
                min_distance=int(np.ceil(1.5*peak_location_radius_9A_theory)))

        # Use the first peak as the maximum
        try:
            peak_location = peaks[0]
        except IndexError as err:
            # No peaks found, take the first maximum found instead
            maxima = find_maxima(eye_mask, mask_center=center,
                    rmin=eyes_rmin,rmax=eyes_rmax)
            peak_location = maxima[0]

        # 2. Use the binary blob method to improve 9A peak location estimate
        eye_roi_binary = np.copy(eye_roi)
        # Calculate percentile
        percentile = np.percentile(eye_roi_binary,eyes_percentile)

        # Binary threshold based on percentile
        eye_roi_binary[eye_roi_binary < percentile] = 0
        eye_roi_binary[eye_roi_binary >= percentile] = 1

        # Now calculate centroid of max blob
        # Now set region of interest for maximum
        eye_max_roi_mask = create_circular_mask(h,w,center=peak_location,rmax=peak_location_radius_9A_theory)
        # Mask out areas outside of eye max roi
        eye_max_roi = np.copy(eye_roi_binary)
        eye_max_roi[~eye_max_roi_mask] = 0
        # Take centroid of this
        eye_max_roi_coordinates = np.array(np.where(eye_max_roi == 1)).T

        blob_centroid = find_centroid(eye_max_roi_coordinates)

        if blob_centroid:
            anchor = blob_centroid
        else:
            anchor = peak_location

        # 3. Calculate the rotation angle of the XRD pattern using result from 9A feature analysis

        # Calculate angle between two points
        angle = get_angle(center, anchor)

        return angle

    def centerize(self, image):
        """
        Move diffraction pattern to the center of the image
        """
        params = self.params
        h = params.get("h")
        w = params.get("w")
        beam_rmax = params.get("beam_rmax")
        rmin = params.get("rmin")
        rmax = params.get("rmax")

        # Calculate array center
        array_center = np.array(image.shape)/2-0.5
        self.array_center = array_center

        # Find center using original image
        center = find_center(image,method="max_centroid",rmax=beam_rmax)
        translation = (array_center[1] - center[1], array_center[0] - center[0])

        # Center the image if need be
        if np.array_equal(center, array_center):
            centered_image = image
        else:
            translation_tform = EuclideanTransform(translation=translation)
            centered_image = warp(image, translation_tform.inverse)

        return centered_image, center

    def centerize_and_rotate(self, image):
        """
        Given an input image, perform the following steps:
        1. determine its center
        2. centerize
        3. find rotation angle
        4. rotate
        """
        params = self.params
        h = params.get("h")
        w = params.get("w")
        beam_rmax = params.get("beam_rmax")
        rmin = params.get("rmin")
        rmax = params.get("rmax")

        # Calculate array center
        array_center = np.array(image.shape)/2-0.5
        self.array_center = array_center

        centered_image, center = self.centerize(image)

        # Find eye rotation using original image
        angle_degrees = self.find_eye_rotation_angle(image, center)

        # Rotate the image
        if np.isclose(angle_degrees, 0):
            centered_rotated_image = centered_image
        else:
            centered_rotated_image = rotate(centered_image, -angle_degrees, preserve_range=True)

        # Returned the centerized and rotated image,
        # the calculated center of the original image, and the rotation angle in degrees
        return centered_rotated_image, center, angle_degrees

    def mask(self, image, center=None, style="both"):
        """
        Mask an image according to style:
        - "beam" means beam mask only
        - "outside" means outer ring mask only
        - "both" means annulus
        """
        params = self.params
        h = params.get("h")
        w = params.get("w")
        rmin = params.get("rmin")
        rmax = params.get("rmax")
        beam_detection = params.get("beam_detection")

        if not center:
            center = self.array_center

        if beam_detection and style in ["both", "beam"]:
            try:
                inflection_point, _, _ = beam_extent(image)
                rmin = inflection_point
            except:
                pass

        # Mask
        if style == "both":
            # Mask out beam and area outside outer ring
            roi_mask = create_circular_mask(h,w,center=center,rmin=rmin,rmax=rmax)
            image[~roi_mask] = 0
        elif style == "beam":
            # Mask out beam
            roi_mask = create_circular_mask(h,w,center=center,rmin=rmin, rmax=h)
            image[~roi_mask] = 0
        elif style == "outside":
            # Mask area outside outer ring
            outside = np.max(image.shape)
            inv_roi_mask = create_circular_mask(h,w,center=center,rmin=rmax,rmax=outside)
            image[inv_roi_mask] = 0

        return image



if __name__ == "__main__":
    """
    Commandline interface

    Directory specifications:
    - Specify the full input_dir and output_dir, or
    - specify the parent_dir, the samples_dir,
      and the program will create a timestamped output directory.

    Parameters specifications:
    - Provide the full path to the params file, or
    - provide a JSON-encoded string of parameters.

    Plans specifications:
    - Provide the plans as a csv (e.g. "plan1,plan2"), or
    - provide the plans in the params (e.g. {... "plans": ["plan1", "plan2"]})


    """
    print("Start preprocessing...")

    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--input_path", default=None,
            help="Path to input files")
    parser.add_argument(
            "--filename", default=None,
            help="Name of file to analyze")
    parser.add_argument(
            "--data_dir", default=None,
            help="Files subdirectory name")
    parser.add_argument(
            "--output_path", default=None,
            help="Path to store preprocessing results")
    parser.add_argument(
            "--params_file", default=None,
            help="The filename with parameters for preprocessing")
    parser.add_argument(
            "--params", default=None,
            help="A JSON-encoded string of parameters for preprocessing")
    parser.add_argument(
            "--plans", default=None,
            help="The plans for preprocessing")
    parser.add_argument(
            "--uniform_filter_size", default=None,
            help="Uniform filter size")
    parser.add_argument(
            "--scaling", default=None,
            help="Plot scaling")
    parser.add_argument(
            "--parallel", default=None, action="store_true",
            help="Flag to run preprocessing in parallel.")

    args = parser.parse_args()

    # Set variables based on input arguments
    # Set directory info
    input_path = args.input_path
    filename = args.filename
    data_dir = args.data_dir
    output_path = args.output_path

    # Get parameters from file or from JSON string commandline argument
    params_file = args.params_file
    params_json = args.params
    if params_file:
        with open(params_file,"r") as params_fp:
            params = json.loads(params_fp.read())
    elif args.params:
        params = json.loads(params_json)
    else:
        raise ValueError("Parameters file or JSON string required.")

    # Get plans from comma-separated values or from parameters
    plans_csv = args.plans
    if plans_csv:
        plans = args.plans.split(",")
    else:
        plans = params.get("plans")
    if not plans:
        raise ValueError("Plans required.")

    # Set uniform filter size
    uniform_filter_size = args.uniform_filter_size
    if not uniform_filter_size:
        uniform_filter_size = 0

    # Set plot scaling
    scaling = args.scaling

    # Check if parallel processing should be used
    parallel = args.parallel

    if parallel:
        # Set the output path
        if not output_path:
            # Set timestamp
            timestr = "%Y%m%dT%H%M%S.%f"
            timestamp = datetime.utcnow().strftime(timestr)
            # Construct the output directory name
            input_dir = os.path.basename(os.path.abspath(input_path))
            output_dir = "preprocessed_{}_{}".format(input_dir, timestamp)
            output_path = os.path.join(input_path, output_dir)

        # Create the output path
        os.makedirs(output_path, exist_ok=True)

        # Get the list of file paths
        input_filepath_list = glob.glob(os.path.join(input_path, "*.txt"))

        # Run all processes in parallel
        process_list = []
        print("Begin parallel preprocessing...")
        for input_filepath in input_filepath_list:
            print(
                    "Preprocessing... {}".format(
                        os.path.basename(input_filepath)))
            # Construct the CLI function call
            function_cli_list = [
                    'python',
                    'eosdxanalysis/preprocessing/preprocess.py',
                    '--input_path',
                    str(input_filepath),
                    '--params_file',
                    str(params_file),
                    '--output_path',
                    str(output_path),
                    ]

            p = subprocess.Popen(function_cli_list)
            process_list.append(p)

        # Wait for all processes to finish
        for p in process_list:
            p.wait()

        print("Done parallel preprocessing.")

    else:
        # Instantiate PreprocessData class
        preprocessor = PreprocessData(
                input_path=input_path, filename=filename, data_dir=data_dir,
                output_path=output_path, params=params)

        # Run preprocessing
        preprocessor.preprocess(plans=plans, mask_style=params.get("crop_style"),
                uniform_filter_size=int(uniform_filter_size), scaling=scaling)

    print("Done preprocessing.")
