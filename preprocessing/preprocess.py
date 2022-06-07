import os
import glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors

from scipy import ndimage
from skimage.filters import threshold_local
import imageio

from preprocessing.center_finding import find_center
from preprocessing.center_finding import find_centroid
from preprocessing.utils import create_circular_mask
from preprocessing.utils import gen_rotation_line
from preprocessing.utils import get_angle
from preprocessing.denoising import filter_strays
from preprocessing.image_processing import centerize
from preprocessing.image_processing import rotate_image
from preprocessing.image_processing import convert_to_cv2_img
from preprocessing.image_processing import crop_image
from preprocessing.image_processing import quadrant_fold


"""
Performs preprocessing pipeline
"""

OUTPUT_MAP = {
        # Maps output style to input plan
        "centered_rotated":"centerize_rotate",
        "quad_folded":"quad_fold",
        "local_thresh_centered_rotated":"local_thresh_centerize_rotate",
        "local_thresh_quad_folded":"local_thresh_quad_fold",
        }

class PreprocessData(object):
    """
    Class to run processing pipeline on data set.
    """

    def __init__(self, filename=None, input_dir=None,
            params={}):
        # Store input and output directories
        self.input_dir = input_dir

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
        self.cache["centered_rotated"] = []
        self.cache["quad_folded"] = []
        self.cache["local_thresh_centered_rotated"] = []
        self.cache["local_thresh_quad_folded"] = []

        return super().__init__()

    def preprocess(self, denoise=False, visualize=False, plans=["centerize_rotate"], mask_style="both"):
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

        # Get filename info
        filenames_fullpaths = self.filenames_fullpaths

        # sample_count
        for filename_fullpath in filenames_fullpaths:
            # Load file
            sample = np.loadtxt(filename_fullpath)
            # Store original in cache
            self.cache["original"].append(sample)

            for plan in plans:

                if plan == "centerize_rotate":
                    # Centerize and rotate
                    centered_rotated_image, center, new_center, angle = self.centerize_and_rotate(sample)
                    # Set output
                    output = centered_rotated_image

                    if mask_style:
                        output = self.mask(centered_rotated_image, style=mask_style)

                    self.cache["centered_rotated"].append(output)

                if plan == "quad_fold":
                    # Centerize and rotate
                    centered_rotated_image, center, new_center, angle = self.centerize_and_rotate(sample)
                    # Quad fold
                    quad_folded_image = quadrant_fold(centered_rotated_image)
                    # Set output
                    output = quad_folded_image

                    if mask_style:
                        output = self.mask(quad_folded_image, style=mask_style)

                    self.cache["quad_folded"].append(output)

                if plan == "local_thresh_centerize_rotate":
                    # Take local threshold
                    local_thresh_image = threshold_local(sample, local_thresh_block_size)
                    # Centerize and rotate
                    local_thresh_centered_rotated_image, center, new_center, angle = self.centerize_and_rotate(local_thresh_image)
                    # Set output
                    output = local_thresh_centered_rotated_image

                    if mask_style:
                        output = self.mask(local_thresh_centered_rotated_image, style=mask_style)

                    self.cache["local_thresh_centered_rotated"].append(output)

                if plan == "local_thresh_quad_fold":
                    # Take local threshold
                    local_thresh_image = threshold_local(sample, local_thresh_block_size)
                    # Centerize and rotate
                    local_thresh_centered_rotated_image, center, new_center, angle = self.centerize_and_rotate(local_thresh_image)
                    # Quad fold
                    local_thresh_quad_folded_image = quadrant_fold(local_thresh_centered_rotated_image)
                    # Set output
                    output = local_thresh_quad_folded_image

                    if mask_style:
                        output = self.mask(local_thresh_quad_folded_image, style=mask_style)

                    self.cache["local_thresh_quad_folded"].append(output)

            if visualize:
                # Recenter image
                local_thresh_image = self.cache["local_thresh"][0]

                # Plot
                fig = plt.figure(dpi=100)
                fig.set_size_inches(4*4,4*1) # x,y
                fig.set_facecolor("white")
                filename = os.path.basename(filename_fullpath)
                fig.suptitle("Preprocessing "+filename)

        #         # Plot histograms
        #         fig.add_subplot(1,2,1)
        #         plt.hist(sample, bins=25, range=(0,100))
        #         plt.title("Original Sample Intensities")

        #         fig.add_subplot(1,2,2)
        #         plt.hist(orig_centered, bins=25, range=(0,100))
        #         plt.title("Centerized Sample Intensities")

        #         print(np.sum(sample),np.sum(orig_centered))

                # Original image
                ax1 = fig.add_subplot(1,4,1)
                plt.imshow(20*np.log10(beam_masked_img+1))
                plt.plot(center[1],center[0],marker='o',color='g')
                ax1.add_artist(mpl.lines.Line2D(*pre_rotation_line,color='g'))
                plt.title("Original [dB+1]")

                # Plot 9A features region of interest
                fig.add_subplot(1,4,2)
                plt.imshow(eye_roi)
    #             plt.imshow(eye_roi_binary)
                plt.plot(initial_max_centroid[1],initial_max_centroid[0],marker='o',color='r')
                plt.plot(eye_max_blob_centroid[1],eye_max_blob_centroid[0],marker='o',color='g')
                plt.title("Feature: 9A Arc Maximas")

                # Final results denoised
                fig.add_subplot(1,4,3)
                plt.imshow(scaled_masked_img)
                plt.plot(256//2,256//2,marker='o',color='g')
                plt.hlines(256//2,256//2-100,256//2+100,color='g')
                plt.title("Rotated Centered Denoised")

                # Original plot preprocessed
                fig.add_subplot(1,4,4)
                plt.imshow(20*np.log10(orig_preprocessed+1))
                plt.title("Rotated Centered Original [dB+1]")

        #         # Quadrant-folded image
        #         fig.add_subplot(1,4,4)
        #         plt.imshow(quad_masked)
        #         plt.title("Quadrant-Folded Original [dB+1]")

                plt.show()

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
        initial_max_centroid = find_center(masked_image,mask_center=center,
                method="max_centroid",rmin=eyes_rmin,rmax=eyes_rmax)

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
        eye_max_blob_centroid = find_centroid(eye_max_roi_coordinates)

        # 3. Calculate the rotation angle of the XRD pattern using result from 9A feature analysis

        # Calculate angle between two points
        angle = get_angle(center, eye_max_blob_centroid)
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
        # Find eye rotation using original image
        angle = self.find_eye_rotation_angle(image, center)
        # Centerize image producing a large image
        centered_image_large, new_center = centerize(image, center)
        # Rotate the centered enlarged image
        centered_rotated_image_large = rotate_image(centered_image_large, new_center, angle)
        # Crop to original size
        centered_rotated_image = crop_image(centered_rotated_image_large, h, w, new_center)
        return centered_rotated_image, center, new_center, angle

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

    def save(self, output_dir=None, output_format="txt", output_style="centered_rotated", rescale=False):
        """
        Save preprocessed image to file
        Inputs:
        - output_dir: Output directory
        - output_format: Output formats are "txt" or "png"
        - output_style: according to preprocessing plan type
        """
        # Make sure cache is not empty
        if self.cache == {}:
            raise ValueError("Image cache is empty, please call preprocess method first.")

        # Check if output style is in list of plans
        plans = self.plans
        if OUTPUT_MAP.get("{}".format(output_style)) not in plans:
            raise ValueError("You did not preprocess with the {} plan.".format(output_style))

        # Get filename info
        filenames_fullpaths = self.filenames_fullpaths
        # Store output directory info
        self.output_dir = output_dir

        # Create output directory only if it does not exist
        os.makedirs(output_dir)

        for idx, filename_fullpath in enumerate(filenames_fullpaths):
            filename = os.path.basename(filename_fullpath)

            # Set the output based on output specifications
            try:
                cache = self.cache["{}".format(output_style)]
            except KeyError as err:
                print("Could not find image cache for style {}.".format(output_style))
                raise err

            # Get the image from the cache
            try:
                output = cache[idx]
            except IndexError as err:
                print("Error accessing image cache.")
                raise err

            if rescale:
                output = convert_to_cv2_img(output)[:,:,0]

            # Save output as text
            if output_format == "txt":
                save_filename = "preprocessed_{}".format(filename)
                save_filename_fullpath = os.path.join(output_dir, save_filename)
                np.savetxt(save_filename_fullpath,output.astype(np.uint16),fmt='%i')

            # Save output as image
            if output_format == "png":
                save_filename = "preprocessed_{}.png".format(filename)
                save_filename_fullpath = os.path.join(output_dir, save_filename)
                imageio.imwrite(save_filename_fullpath, output.astype(np.uint16))
