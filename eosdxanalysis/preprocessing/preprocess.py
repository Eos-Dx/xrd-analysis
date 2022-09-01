import os
import glob
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors

from scipy import ndimage
from skimage.filters import threshold_local
from skimage.transform import EuclideanTransform
from skimage.transform import warp
from skimage.transform import rotate

from eosdxanalysis.preprocessing.center_finding import find_center
from eosdxanalysis.preprocessing.center_finding import find_centroid
from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.preprocessing.utils import gen_rotation_line
from eosdxanalysis.preprocessing.utils import get_angle
from eosdxanalysis.preprocessing.utils import find_maxima
from eosdxanalysis.preprocessing.denoising import filter_strays
from eosdxanalysis.preprocessing.image_processing import convert_to_cv2_img
from eosdxanalysis.preprocessing.image_processing import crop_image
from eosdxanalysis.preprocessing.image_processing import quadrant_fold


"""
Performs preprocessing pipeline
"""

ABBREVIATIONS = {
        # output style: output style abbreviation
        "centered_rotated": "CR",
        "centered_rotated_quad_folded": "CRQF",
        "local_thresh_centered_rotated": "LTCR",
        "local_thresh_centered_rotated_quad_folded": "LTCRQF",
        }

OUTPUT_MAP = {
        # Maps output style to input plan
        "centered_rotated":"centerize_rotate",
        "centered_rotated_quad_folded":"centerize_rotate_quad_fold",
        "local_thresh_centered_rotated":"local_thresh_centerize_rotate",
        "local_thresh_centered_rotated_quad_folded":"local_thresh_centerize_rotate_quad_fold",
        }

INVERSE_OUTPUT_MAP = {
        # Maps preprocessing plan to output style
        "centerize_rotate":"centered_rotated",
        "centerize_rotate_quad_fold":"centered_rotated_quad_folded",
        "local_thresh_centerize_rotate":"local_thresh_centered_rotated",
        "local_thresh_centerize_rotate_quad_fold":"local_thresh_centered_rotated_quad_folded",
        }


class PreprocessData(object):
    """
    Class to run processing pipeline on data set.
    """

    def __init__(self, filename=None, parent_dir=None, samples_dir=None,
            input_dir=None, output_dir=None, params={}):
        # Store input and output directories
        self.parent_dir = parent_dir
        self.samples_dir = samples_dir
        self.input_dir = input_dir
        self.output_dir = output_dir

        # If we are provided with a single image, set its full path
        if filename != None and input_dir != None:
            # Collect full file fullpath into a single-element list
            filenames_fullpaths = [os.path.join(input_dir, filename)]

        # Else we are provided with an input directory
        elif filename == None and input_dir != None:
            # Get sorted list of filenames
            filenames_fullpaths = glob.glob(os.path.join(input_dir,"*.txt"))
            filenames_fullpaths.sort()

        # Store filenames (these are full paths)
        self.filenames_fullpaths = filenames_fullpaths

        # Load data from files and store
        # Vectorization/parallelization
        # df_list = [pd.read_csv(datafile) for datafile in filenames_fullpaths]
        # self.df_list = df_list

        # Store parameters
        self.params = params

        self.cache = {}
        self.cache["original"] = []
        self.cache["local_thresh"] = []
        for plan in OUTPUT_MAP.keys():
            self.cache[plan] = []

        return super().__init__()

    def preprocess(self, denoise=False, plans=["centerize_rotate"],
                mask_style="both", uniform_filter_size=None):
        """
        Run all preprocessing steps

        Plan options include:
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
        self.plans = plans
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

        # Set mask style from params if crop_style is set
        mask_style = params.get("crop_style", mask_style)

        # Get filename info
        filenames_fullpaths = self.filenames_fullpaths

        # Set timestamp
        timestr = "%Y%m%dT%H%M%S.%f"
        timestamp = datetime.utcnow().strftime(timestr)

        # Store output directory info
        if not self.output_dir:
            # Create output directory if it does not exist
            if samples_dir:
                output_dir_name = "preprocessed_{}_{}".format(
                                        samples_dir, timestamp)
            else:
                output_dir_name = "preprocessed_{}".format(timestamp)
            output_dir = os.path.join(parent_dir, output_dir_name)
            os.makedirs(output_dir, exist_ok=True)

        print("Saving to", output_dir, "...")

        # Write params to file
        with open(os.path.join(output_dir,"params.txt"),"w") as paramsfile:
            paramsfile.write(json.dumps(params,indent=4))


        # Loop over plans
        for plan in plans:

            output_style = INVERSE_OUTPUT_MAP.get(plan)
            output_style_abbreviation = ABBREVIATIONS.get(output_style)

            # Create output directory for plan and output format
            plan_output_dir = os.path.join(output_dir, output_style)
            os.makedirs(plan_output_dir, exist_ok=True)

            # Loop over files
            for filename_fullpath in filenames_fullpaths:

                # Load file
                sample = np.loadtxt(filename_fullpath)

                filename = os.path.basename(filename_fullpath)

                # Set the output based on output specifications
                try:
                    cache = self.cache["{}".format(output_style)]
                except KeyError as err:
                    print("Could not find image cache for style {}.".format(output_style))
                    raise err


                if plan == "centerize_rotate":
                    # Centerize and rotate
                    centered_rotated_image, center, angle = self.centerize_and_rotate(sample)
                    # Set output
                    output = centered_rotated_image

                if plan == "centerize_rotate_quad_fold":
                    # Centerize and rotate
                    centered_rotated_image, center, angle = self.centerize_and_rotate(sample)
                    # Quad fold
                    centered_rotated_quad_folded_image = quadrant_fold(centered_rotated_image)
                    # Set output
                    output = centered_rotated_quad_folded_image

                if plan == "local_thresh_centerize_rotate":
                    # Take local threshold
                    local_thresh_image = threshold_local(sample, local_thresh_block_size)
                    # Centerize and rotate
                    local_thresh_centered_rotated_image, center, angle = self.centerize_and_rotate(local_thresh_image)
                    # Set output
                    output = local_thresh_centered_rotated_image

                if plan == "local_thresh_centerize_rotate_quad_fold":
                    # Take local threshold
                    local_thresh_image = threshold_local(sample, local_thresh_block_size)
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
                    output = self.mask(output, style=mask_style)

                # Save the file
                save_filename = "{}_{}".format(output_style_abbreviation,
                                            filename)
                save_filename_fullpath = os.path.join(plan_output_dir,
                                            save_filename)
                np.savetxt(save_filename_fullpath,
                                np.round(output).astype(np.uint16), fmt='%i')

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

        # 1. Find the 9A arc maxima features

        # Perform local thresholding on original
        local_thresh_image = threshold_local(image, local_thresh_block_size)

        # Add local threshold image to cache
        self.cache["local_thresh"].append(local_thresh_image)

        # Apply circular mask as a digital beamstop to center
        beam_mask = create_circular_mask(h,w,center=center,rmin=rmin,rmax=rmax)
        # masked_image = np.copy(local_thresh_image)
        masked_image = np.copy(image)
        masked_image[~beam_mask] = 0

        # Get the max_centroid as a starting guess
        maxima = find_maxima(masked_image,mask_center=center,
                rmin=eyes_rmin,rmax=eyes_rmax)

        # Take the first maximum
        initial_max_centroid = maxima[0]

        # 2. Use percentile thresholding to convert 9A arc maxima features to blobs

        # Create mask for eye region
        eye_mask = create_circular_mask(h,w,center=center,rmin=eyes_rmin,rmax=eyes_rmax)
        # Use local threshold results
        eye_roi = np.copy(local_thresh_image)
        eye_roi[~eye_mask] = 0
        eye_roi_binary = np.copy(eye_roi)
        # Calculate percentile
        percentile = np.percentile(eye_roi_binary,eyes_percentile)

        # Binary threshold based on percentile
        eye_roi_binary[eye_roi_binary < percentile] = 0
        eye_roi_binary[eye_roi_binary >= percentile] = 1

        # Now calculate centroid of max blob

        # Now set region of interest for maximum
        eye_max_roi_mask = create_circular_mask(h,w,center=initial_max_centroid,rmax=eyes_blob_rmax)
        # Mask out areas outside of eye max roi
        eye_max_roi = np.copy(eye_roi_binary)
        eye_max_roi[~eye_max_roi_mask] = 0
        # Take centroid of this
        eye_max_roi_coordinates = np.array(np.where(eye_max_roi == 1)).T

        blob_centroid = None
        if np.any(eye_max_roi_coordinates):
            blob_centroid = find_centroid(eye_max_roi_coordinates)
            centroid = blob_centroid

        # If we do not get a result, use the initial eye max centroid
        if not blob_centroid:
            centroid = initial_max_centroid

        # 3. Calculate the rotation angle of the XRD pattern using result from 9A feature analysis

        # Calculate angle between two points
        angle = get_angle(center, centroid)

        return angle

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

        # Find center using original image
        center = find_center(image,method="max_centroid",rmax=beam_rmax)
        array_center = (image.shape[0]/2-0.5, image.shape[1]/2-0.5)
        # Find eye rotation using original image
        angle_degrees = self.find_eye_rotation_angle(image, center)
        translation = (array_center[1] - center[1], array_center[0] - center[0])

        # Center the image if need be
        if np.array_equal(center, array_center):
            centered_image = image
        else:
            translation_tform = EuclideanTransform(translation=translation)
            centered_image = warp(image, translation_tform.inverse)

        # Rotate the image
        if np.isclose(angle_degrees, 0):
            centered_rotated_image = centered_image
        else:
            centered_rotated_image = rotate(centered_image, -angle_degrees, preserve_range=True)

        # Centerize the image
        return centered_rotated_image, center, angle_degrees

    def mask(self, image, style="both"):
        """
        Mask an image according to style:
        - "beam" means beam mask only
        - "outer" means outer ring mask only
        - "both" means annulus
        """
        params = self.params
        h = params.get("h")
        w = params.get("w")
        rmin = params.get("rmin")
        rmax = params.get("rmax")

        # Mask
        if style == "both":
            # Mask out beam and area outside outer ring
            roi_mask = create_circular_mask(h,w,rmin=rmin,rmax=rmax)
            image[~roi_mask] = 0
        if style == "beam":
            # Mask out beam
            roi_mask = create_circular_mask(h,w,rmin=rmin, rmax=h)
            image[~roi_mask] = 0
        if style == "outside":
            # Mask area outside outer ring
            inv_roi_mask = create_circular_mask(h,w,rmin=rmax)
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
            "--parent_dir", default=None,
            help="The parent directory to analyze")
    parser.add_argument(
            "--samples_dir", default=None,
            help="The subdirectory name of samples to analyze")
    parser.add_argument(
            "--input_dir", default=None,
            help="The files directory to analyze")
    parser.add_argument(
            "--output_dir", default=None,
            help="The directory to preprocessing results")
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

    args = parser.parse_args()

    # Set variables based on input arguments
    # Set directory info
    parent_dir = args.parent_dir
    samples_dir = args.samples_dir
    input_dir = args.input_dir
    output_dir = args.output_dir

    # If parent_dir and samples_dir are set, and input_dir and output_dir
    # are empty, then set input_dir and output_dir appropriately
    if parent_dir and samples_dir and not input_dir and not output_dir:
        input_dir = os.path.join(parent_dir, samples_dir)
    # If parent_dir and samples_dir are not set, but input_dir and output_dir
    # are set, then just use those
    elif input_dir and output_dir:
        pass
    elif parent_dir and samples_dir and output_dir:
        input_dir = os.path.join(parent_dir, samples_dir)
    else:
        raise ValueError("Must specify parent_dir and samples_dir or input_dir")

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

    # Instantiate PreprocessData class
    preprocessor = PreprocessData(input_dir=input_dir, output_dir=output_dir,
            parent_dir=parent_dir, samples_dir=samples_dir, params=params)

    # Run preprocessing
    preprocessor.preprocess(plans=plans, mask_style=params.get("crop_style"),
            uniform_filter_size=uniform_filter_size)

    print("Done preprocessing.")
