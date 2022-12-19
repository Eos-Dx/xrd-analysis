"""
Feature extraction functions
"""
import os
import glob
import argparse
import json

import numpy as np
import pandas as pd

from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.preprocessing.image_processing import bright_pixel_count

from eosdxanalysis.calibration.utils import DiffractionUnitsConversion


class FeatureExtraction(object):
    """
    Class to handle feature extraction
    """

    def __init__(
            self, image, source_wavelength=None, pixel_length=None,
            sample_to_detector_distance=None):

        """
        Initializes the FeatureExtraction class with image path and
        machine parameters.

        Parameters
        ----------

        image : (n,2)-ndarray
            Image data

        source_wavelength: number

        pixel_length : number

        sample_to_detector_distance : number

        Notes
        -----
        See ``eosdxanalysis.calibration.utils.DiffractionUnitsConversion`` for
        explanation of machine parameters.

        """
        # Store input image
        self.image = image

        # Store machine parameters
        self.source_wavelength = source_wavelength
        self.pixel_length = pixel_length
        self.sample_to_detector_distance = sample_to_detector_distance

        return super().__init__()

    def feature_max_intensity(self):
        """
        Computes the max intensity of the image

        Parameters
        ----------

        image : ndarray
            The measurement image data

        Output
        ------

        max_intensity : number
            The maximum intensity

        """
        # Reference the stored image
        image = self.image

        # Compute the maximum image intensity
        max_intensity = np.max(image)

        return max_intensity

    def feature_image_intensity(self):
        """
        Computes the intensity of the image

        Parameters
        ----------

        image : ndarray
            The measurement image data

        Output
        ------

        image_intensity : number
            The total intensity

        """
        # Reference the stored image
        image = self.image

        # Compute the total image intensity
        image_intensity = np.sum(image)

        return image_intensity

    def feature_bright_pixel_count(self, threshold=0.75):
        """
        Computes the intensity of the image

        Parameters
        ----------

        threshold : float
            The brightness threshold

        Output
        ------

        bright_pixels : number
            The number of bright pixels

        """
        # Reference the stored image
        image = self.image

        # Compute the bright pixel count
        bright_pixels = bright_pixel_count(image, qmin=threshold)

        return bright_pixels

    def feature_annulus_intensity(self, center=None, rmin=None, rmax=None):
        """
        Computes the total intensity of an annulus

        Parameters
        ----------

        center : 2-tuple (number)
            Row and column location of the annulus center

        rmin : number
            Minimum annulus radius in pixel units

        rmax : number
            Maximum annulus radius in pixel units

        Output
        ------

        annulus_intensity : number
            The intensity of the annulus
        """
        # Reference the stored image
        image = self.image

        # Get the image shape
        shape = image.shape

        # Create a mask for the annulus
        annulus_mask = create_circular_mask(shape[0], shape[1], center=center,
                rmin=rmin, rmax=rmax)

        # Compute the annulus intensity
        annulus_intensity = np.sum(image[annulus_mask])

        return annulus_intensity

    def feature_annulus_intensity_angstroms(
            self, pixel_length=None, center=None, amin=None, amax=None):
        """
        Calculate the intensity of an annulus specified by start and end radii
        in Angstrom units.

        Parameters
        ----------

        distance_to_detector : number
            Distance from sample to detector

        center : 2-tuple (number)
            Row and column location of the annulus center

        amin : number
            Minimum annulus angstrom in pixel units

        amax : number
            Maximum annulus angstrom in pixel units

        Output
        ------

        annulus_intensity : number
            The intensity of the annulus


        Notes
        -----

        ``rmin`` and ``rmax`` are swapped from ``amin`` and ``amax`` since we
        are in reciprocal space.

        """
        # Get machine parameters
        source_wavelength = self.source_wavelength
        sample_to_detector_distance = self.sample_to_detector_distance

        # Check if required machine parameters were provided
        if not all([self.source_wavelength, self.sample_to_detector_distance]):
            raise ValueError("You must initialize the units class with machine"
                    " parameters for this method!")

        # Reference the stored image
        image = self.image

        # Get the image shape
        shape = image.shape

        # Initialize the units class
        units_class = DiffractionUnitsConversion(
                source_wavelength=source_wavelength, pixel_length=pixel_length,
                sample_to_detector_distance=sample_to_detector_distance)

        # Set rmin to the larger angstrom spacing (reciprocal space)
        rmin = units_class.bragg_peak_pixel_location_from_molecular_spacing(
                amax)
        # Set rmax to the smaller angstrom spacing (repciprocal space)
        rmax = units_class.bragg_peak_pixel_location_from_molecular_spacing(
                amin)

        # Create a mask for the annulus
        annulus_mask = create_circular_mask(shape[0], shape[1], center=center,
                rmin=rmin, rmax=rmax)

        # Compute the annulus intensity
        annulus_intensity = np.sum(image[annulus_mask])

        return annulus_intensity

    def feature_sector_intensity(
            self, center=None, rmin=None, rmax=None, theta_min=-np.pi/4,
            theta_max=np.pi/4):
        """
        Computes the total intensity of a sector

        Parameters
        ----------

        center : 2-tuple (number)
            Row and column location of the annulus center

        rmin : number
            Minimum annulus radius in pixel units

        rmax : number
            Maximum annulus radius in pixel units

        theta_min : number
            Start angle of sector in radians. 0 degrees is x > 0 axis. The
            direction of positive theta is counter-clockwise.

        theta_max : number
            End angle of sector in radians. 0 degrees is x > 0 axis. The
            direction of positive theta is counter-clockwise.

        Output
        ------

        sector_intensity : number
            The intensity of the sector

        Notes
        -----
        Angles are measured in radians. The x > 0 axis is zero degrees. The
        direction of positive theta is counter-clockwise.
        """
        if theta_min < -np.pi or theta_max > np.pi:
            raise ValueError(
                    "Sector angle bounds must be between -pi and +pi.")

        # Reference the stored image
        image = self.image

        # Get the image shape
        shape = image.shape

        # Create a mask for the annulus
        annulus_mask = create_circular_mask(shape[0], shape[1], center=center,
                rmin=rmin, rmax=rmax)

        # Generate a meshgrid the same size as the image
        x_end = shape[1]/2 - 0.5
        x_start = -x_end
        y_end = x_start
        y_start = x_end
        YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
        TT = np.arctan2(YY, XX)

        sector_indices = (TT > theta_min) & (TT < theta_max) & annulus_mask

        sector_intensity = np.sum(annulus_mask[sector_indices])

        return sector_intensity

    def feature_sector_intensity_angstroms(
            self, pixel_length=None, center=None, amin=None, amax=None,
            theta_min=-np.pi/4, theta_max=np.pi/4):
        """
        Computes the total intensity of a sector

        Parameters
        ----------

        center : 2-tuple (number)
            Row and column location of the annulus center

        amin : number
            Minimum annulus radius in pixel units

        amax : number
            Maximum annulus radius in pixel units

        theta_min : number
            Start angle of sector in radians. 0 degrees is x > 0 axis. The
            direction of positive theta is counter-clockwise.

        theta_max : number
            End angle of sector in radians. 0 degrees is x > 0 axis. The
            direction of positive theta is counter-clockwise.

        Output
        ------

        sector_intensity : number
            The intensity of the sector

        Notes
        -----
        Angles are measured in radians. The x > 0 axis is zero degrees. The
        direction of positive theta is counter-clockwise.
        """
        # Get machine parameters
        source_wavelength = self.source_wavelength
        sample_to_detector_distance = self.sample_to_detector_distance

        # Check if required machine parameters were provided
        if not all([self.source_wavelength, self.sample_to_detector_distance]):
            raise ValueError("You must initialize the units class with machine"
                    " parameters for this method!")

        # Reference the stored image
        image = self.image

        # Get the image shape
        shape = image.shape

        # Initialize the units class
        units_class = DiffractionUnitsConversion(
                source_wavelength=source_wavelength, pixel_length=pixel_length,
                sample_to_detector_distance=sample_to_detector_distance)

        # Set rmin to the larger angstrom spacing (reciprocal space)
        rmin = units_class.bragg_peak_pixel_location_from_molecular_spacing(
                amax)
        # Set rmax to the smaller angstrom spacing (repciprocal space)
        rmax = units_class.bragg_peak_pixel_location_from_molecular_spacing(
                amin)

        sector_intensity = self.feature_sector_intensity(
                center=center, rmin=rmin, rmax=rmin, theta_min=theta_min,
                theta_max=theta_max)

        return sector_intensity

    def feature_sector_intensity_equator_pair(
            self, center=None, rmin=None, rmax=None,
            sector_angle=None):
        """
        Computes the total intensity of a pair of equator sectors

        Parameters
        ----------

        center : 2-tuple (number)
            Row and column location of the annulus center

        rmin : number
            Minimum annulus radius in pixel units

        rmax : number
            Maximum annulus radius in pixel units

        sector_angle : number
            Angle of sector in radians. Sector is symmetric about equator.
            0 degrees is x > 0 axis. The direction of positive theta is
            counter-clockwise.

        Output
        ------

        equator_pair_intensity : number
            The combined intensity of the equator sector pairs

        Notes
        -----
        Angles are measured in radians. The x > 0 axis is zero degrees. The
        direction of positive theta is counter-clockwise.
        """
        if sector_angle < 0 or sector_angle > np.pi:
            raise ValueError(
                    "Sector angle bounds must be between 0 and +pi.")

        # Reference the stored image
        image = self.image

        # Get the image shape
        shape = image.shape

        # Create a mask for the annulus
        annulus_mask = create_circular_mask(shape[0], shape[1], center=center,
                rmin=rmin, rmax=rmax)

        # Generate a meshgrid the same size as the image
        x_end = shape[1]/2 - 0.5
        x_start = -x_end
        y_end = x_start
        y_start = x_end
        YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
        TT = np.arctan2(YY, XX)

        # Calculate sector start and end angles based on symmetric sector angle
        theta_min = -sector_angle/2
        theta_max = sector_angle/2

        # Get right sector indices
        right_sector_indices = \
                (TT > theta_min) & (TT < theta_max) & annulus_mask
        # Get left sector indices based on left-right symmetry
        left_sector_indices = np.fliplr(right_sector_indices)

        # Compute right and left sector intensities
        right_sector_intensity = np.sum(image[right_sector_indices])
        left_sector_intensity = np.sum(image[left_sector_indices])

        # Compute total equator pair intensity
        equator_pair_intensity = right_sector_intensity + left_sector_intensity

        return equator_pair_intensity

    def feature_sector_intensity_equator_pair_angstroms(
            self, pixel_length=None, center=None, amin=None, amax=None,
            sector_angle=None):
        """
        Computes the total intensity of a pair of equator sectors specified by
        angstrom bounds.

        Parameters
        ----------

        center : 2-tuple (number)
            Row and column location of the annulus center

        amin : number
            Minimum annulus radius in pixel units

        amax : number
            Maximum annulus radius in pixel units

        sector_angle : number
            Angle of sector in radians. Sector is symmetric about equator.
            0 degrees is x > 0 axis. The direction of positive theta is
            counter-clockwise.

        Output
        ------

        sector_intensity : number
            The intensity of the sector

        Notes
        -----
        Angles are measured in radians. The x > 0 axis is zero degrees. The
        direction of positive theta is counter-clockwise.
        """
        if sector_angle < 0 or sector_angle > np.pi:
            raise ValueError(
                    "Sector angle bounds must be between 0 and +pi.")

        # Reference the stored image
        image = self.image

        # Get the image shape
        shape = image.shape

        # Get machine parameters
        source_wavelength = self.source_wavelength
        sample_to_detector_distance = self.sample_to_detector_distance

        # Check if required machine parameters were provided
        if not all([self.source_wavelength, self.sample_to_detector_distance]):
            raise ValueError("You must initialize the units class with machine"
                    " parameters for this method!")

        # Initialize the units class
        units_class = DiffractionUnitsConversion(
                source_wavelength=source_wavelength, pixel_length=pixel_length,
                sample_to_detector_distance=sample_to_detector_distance)

        # Set rmin to the larger angstrom spacing (reciprocal space)
        rmin = units_class.bragg_peak_pixel_location_from_molecular_spacing(
                amax)
        # Set rmax to the smaller angstrom spacing (repciprocal space)
        rmax = units_class.bragg_peak_pixel_location_from_molecular_spacing(
                amin)

        # Create a mask for the annulus
        annulus_mask = create_circular_mask(shape[0], shape[1], center=center,
                rmin=rmin, rmax=rmax)

        # Generate a meshgrid the same size as the image
        x_end = shape[1]/2 - 0.5
        x_start = -x_end
        y_end = x_start
        y_start = x_end
        YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
        TT = np.arctan2(YY, XX)

        # Calculate sector start and end angles based on symmetric sector angle
        theta_min = -sector_angle/2
        theta_max = sector_angle/2

        # Get right sector indices
        right_sector_indices = \
                (TT > theta_min) & (TT < theta_max) & annulus_mask
        # Get left sector indices based on left-right symmetry
        left_sector_indices = np.fliplr(right_sector_indices)

        # Compute right and left sector intensities
        right_sector_intensity = np.sum(image[right_sector_indices])
        left_sector_intensity = np.sum(image[left_sector_indices])

        # Compute total equator pair intensity
        equator_pair_intensity = right_sector_intensity + left_sector_intensity

        return equator_pair_intensity

    def feature_sector_intensity_meridian_pair(
            self, center=None, rmin=None, rmax=None,
            sector_angle=None):
        """
        Computes the total intensity of a pair of meridian sectors

        Parameters
        ----------

        center : 2-tuple (number)
            Row and column location of the annulus center

        rmin : number
            Minimum annulus radius in pixel units

        rmax : number
            Maximum annulus radius in pixel units

        sector_angle : number
            Angle of sector in radians. Sector is symmetric about equator.
            0 degrees is x > 0 axis. The direction of positive theta is
            counter-clockwise.

        Output
        ------

        sector_intensity : number
            The intensity of the sector

        Notes
        -----
        Angles are measured in radians. The x > 0 axis is zero degrees. The
        direction of positive theta is counter-clockwise.
        """
        if sector_angle < 0 or sector_angle > np.pi:
            raise ValueError(
                    "Sector angle bounds must be between 0 and +pi.")

        # Reference the stored image
        image = self.image

        # Get the image shape
        shape = image.shape

        # Create a mask for the annulus
        annulus_mask = create_circular_mask(shape[0], shape[1], center=center,
                rmin=rmin, rmax=rmax)

        # Generate a meshgrid the same size as the image
        x_end = shape[1]/2 - 0.5
        x_start = -x_end
        y_end = x_start
        y_start = x_end
        YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
        TT = np.arctan2(YY, XX)

        # Calculate sector start and end angles based on symmetric sector angle
        theta_min = -sector_angle/2 + np.pi/2
        theta_max = sector_angle/2 + np.pi/2

        # Get top sector indices
        top_sector_indices = \
                (TT > theta_min) & (TT < theta_max) & annulus_mask
        # Get bottom sector indices based on bottom-top symmetry
        bottom_sector_indices = np.flipud(top_sector_indices)

        # Compute top and bottom sector intensities
        top_sector_intensity = np.sum(image[top_sector_indices])
        bottom_sector_intensity = np.sum(image[bottom_sector_indices])

        # Compute total equator pair intensity
        equator_pair_intensity = top_sector_intensity + bottom_sector_intensity

        return equator_pair_intensity

    def feature_sector_intensity_meridian_pair_angstroms(
            self, pixel_length=None, center=None, amin=None, amax=None,
            sector_angle=None):
        """
        Computes the total intensity of a pair of meridian sectors specified by
        angstrom bounds.

        Parameters
        ----------

        center : 2-tuple (number)
            Row and column location of the annulus center

        amin : number
            Minimum annulus radius in pixel units

        amax : number
            Maximum annulus radius in pixel units

        sector_angle : number
            Angle of sector in radians. Sector is symmetric about equator.
            0 degrees is x > 0 axis. The direction of positive theta is
            counter-clockwise.

        Output
        ------

        sector_intensity : number
            The intensity of the sector

        Notes
        -----
        Angles are measured in radians. The x > 0 axis is zero degrees. The
        direction of positive theta is counter-clockwise.
        """
        if sector_angle < 0 or sector_angle > np.pi:
            raise ValueError(
                    "Sector angle bounds must be between 0 and +pi.")

        # Reference the stored image
        image = self.image

        # Get the image shape
        shape = image.shape

        # Get machine parameters
        source_wavelength = self.source_wavelength
        sample_to_detector_distance = self.sample_to_detector_distance

        # Check if required machine parameters were provided
        if not all([self.source_wavelength, self.sample_to_detector_distance]):
            raise ValueError("You must initialize the units class with machine"
                    " parameters for this method!")

        # Initialize the units class
        units_class = DiffractionUnitsConversion(
                source_wavelength=source_wavelength, pixel_length=pixel_length,
                sample_to_detector_distance=sample_to_detector_distance)

        # Set rmin to the larger angstrom spacing (reciprocal space)
        rmin = units_class.bragg_peak_pixel_location_from_molecular_spacing(
                amax)
        # Set rmax to the smaller angstrom spacing (repciprocal space)
        rmax = units_class.bragg_peak_pixel_location_from_molecular_spacing(
                amin)

        # Create a mask for the annulus
        annulus_mask = create_circular_mask(shape[0], shape[1], center=center,
                rmin=rmin, rmax=rmax)

        # Generate a meshgrid the same size as the image
        x_end = shape[1]/2 - 0.5
        x_start = -x_end
        y_end = x_start
        y_start = x_end
        YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
        TT = np.arctan2(YY, XX)

        # Calculate sector start and end angles based on symmetric sector angle
        theta_min = -sector_angle/2 + np.pi/2
        theta_max = sector_angle/2 + np.pi/2

        # Get top sector indices
        top_sector_indices = \
                (TT > theta_min) & (TT < theta_max) & annulus_mask
        # Get bottom sector indices based on bottom-top symmetry
        bottom_sector_indices = np.fliplr(top_sector_indices)

        # Compute top and bottom sector intensities
        top_sector_intensity = np.sum(image[top_sector_indices])
        bottom_sector_intensity = np.sum(image[bottom_sector_indices])

        # Compute total equator pair intensity
        meridian_pair_intensity = top_sector_intensity + bottom_sector_intensity

        return meridian_pair_intensity

def feature_extraction(input_path, output_filepath, params):
    """
    Extracts features on dataset and writes output to csv file.


    Parameters
    ----------

    input_path : str
        Path to centered, rotated data

    output_filepath : str
        Saves extracted features to this file.

    params : str
        JSON-encoded string specifying machine parameters and features to
        extract. See Notes below.

    Notes
    -----
    Assume we are using angstroms for annulus and sector bounds.

    Params JSON-encoded file example, extracted three features:
        1. total intensity
        2. 9 A annulus intensity
        3. 9 A equator sector intensities

        {
            "machine_parameters": [
                "source_wavelength": 1.5418E-10,
                "pixel_length": 55E-6,
                "sample_to_detector_distance": 10E-3
            ],  
            "features": [
                "cropped_intensity": true,
                "annulus_intensity_9A": [
                    8.8E-10,
                    10.8E-10
                ],  
                "sector_intensity_equator_9A": [
                    8.8E-10,
                    10.8E-10,
                    -0.7853981633974483,
                    0.7853981633974483
                ]   
            ]   
        }
    """
    # Load machine parameters
    machine_parameters = params["machine_parameters"]
    source_wavelength = machine_parameters["source_wavelength"]
    pixel_length = machine_parameters["pixel_length"]
    sample_to_detector_distance = machine_parameters[
            "sample_to_detector_distance"]

    # Get filepath list
    filepath_list = glob.glob(os.path.join(input_path, "*.txt"))
    # Sort files list
    filepath_list.sort()

    # Get list of features to extract
    features = params["features"]
    feature_list = list(features.keys())

    # Create dataframe to collect extracted features
    columns = ["Filename"] + feature_list
    df = pd.DataFrame(data={}, columns=columns)

    # Loop over files list
    for filepath in filepath_list:
        filename = os.path.basename(filepath)
        image = np.loadtxt(filepath, dtype=np.uint32)

        # Initialize feature extraction class
        feature_extraction = FeatureExtraction(
                image, source_wavelength=source_wavelength,
                sample_to_detector_distance=sample_to_detector_distance)

        extracted_feature_list = []
        for feature in feature_list:
            if "max_intensity" in feature:
                # Compute max intensity
                max_intensity = feature_extraction.feature_max_intensity()
                extracted_feature_list.append(max_intensity)
            if "cropped_intensity" in feature:
                # Compute cropped intensity
                cropped_intensity = feature_extraction.feature_image_intensity()
                extracted_feature_list.append(cropped_intensity)
            if "bright_pixel_count" in feature:
                # Compute bright pixel count
                bright_pixel_count_threshold = features[feature][0]
                bright_pixel_count = feature_extraction.feature_bright_pixel_count(
                        threshold=bright_pixel_count_threshold)
                extracted_feature_list.append(bright_pixel_count)
            elif "annulus_intensity" in feature:
                # Compute annulus intensity
                annulus_bounds = features[feature]
                amin, amax = annulus_bounds
                annulus_intensity = \
                        feature_extraction.feature_annulus_intensity_angstroms(
                            pixel_length=pixel_length, amin=amin, amax=amax)
                extracted_feature_list.append(annulus_intensity)
            elif "sector_intensity_equator_pair" in feature:
                # Compute sector intensity
                sector_bounds = features[feature]
                amin, amax, sector_angle = sector_bounds
                sector_intensity = \
                        feature_extraction.feature_sector_intensity_equator_pair_angstroms(
                            pixel_length, amin=amin, amax=amax,
                            sector_angle=sector_angle)
                extracted_feature_list.append(sector_intensity)
            elif "sector_intensity_meridian_pair" in feature:
                # Compute sector intensity
                sector_bounds = features[feature]
                amin, amax, sector_angle = sector_bounds
                sector_intensity = \
                        feature_extraction.feature_sector_intensity_meridian_pair_angstroms(
                            pixel_length, amin=amin, amax=amax,
                            sector_angle=sector_angle)
                extracted_feature_list.append(sector_intensity)

        # Add extracted features to dataframe
        df.loc[len(df.index)+1] = [filename] + extracted_feature_list

    # Save dataframe to csv
    df.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    """
    Run feature extraction on a file or entire folder. Provide centered and
    rotated images.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--input_path", default=None, required=True,
            help="The path containing raw files to perform feature extraction"
            " on")
    parser.add_argument(
            "--output_filepath", default=None, required=True,
            help="The output filepath to store extracted features")
    parser.add_argument(
            "--params_filepath", default=None, required=True,
           help="Path to file with parameters for feature extraction")

    args = parser.parse_args()

    input_path = args.input_path
    output_filepath = args.output_filepath

    # Get parameters from file or from JSON string commandline argument
    params_filepath = args.params_filepath
    with open(params_filepath,"r") as params_fp:
        params = json.loads(params_fp.read())

    # Run feature extraction
    feature_extraction(input_path, output_filepath, params)
