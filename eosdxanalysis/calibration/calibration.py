"""
Code to calibrate X-ray diffraction setup using calibration samples data
"""
import numpy as np

from skimage.transform import warp_polar

from eosdxanalysis.calibration.materials import q_peaks_ref_dict

from eosdxanalysis.preprocessing.center_finding import find_center
from eosdxanalysis.preprocessing.image_processing import centerize
from eosdxanalysis.preprocessing.image_processing import unwarp_polar


class Calibration(object):
    """
    Calibration class to perform calibration for a set of calibration images
    """

    def __init__(self, calibration_material, wavelen=1.5418):
        """
        Initialize Calibration class
        """
        # Store source wavelength
        self.wavelen = wavelen

        # Store calibration material name
        self.calibration_material = calibration_material

        # Look up calibration material q-peaks reference data
        try:
            self.q_peaks_ref = q_peaks_ref_dict[calibration_material]
        except KeyError as err:
            print("Calibration material {} not found!".format(
                                            calibration_material))
        return super().__init__()

    def single_sample_detector_distance(self, image, r_max=None):
        """
        Calculate the sample-to-detector distance for a single sample

        Steps performed:
        - Centerize the image
        - Warp to polar
        - Convert to 1D intensity vs. radius (pixel)
        - For each q-peak reference value, calculate
          the sample-to-detector distance
        - Return the mean, standard deviation, and values

        """
        # Centerize the image
        center = find_center(image)
        centered_image, new_center = centerize(image, center)

        # Warp to polar
        # Set a maximum radius which we are interested in
        final_r = int(r_max) if r_max else None
        # Double the size of the output image for better interpolation
        output_shape = (2*centered_image.shape[0], 2*centered_image.shape[1])
        polar_image = warp_polar(centered_image, radius=final_r,
                                output_shape=output_shape, preserve_range=True)

        # Convert to 1D intensity vs. radius (pixel) and rescale by shape
        radial_intensity = np.sum(polar_image, axis=0)/output_shape[0]
        # Set up radius linspace, where r is in pixels
        r_space = np.linspace(0, final_r, len(radial_intensity))

        # Perform Gaussian peak-finding on the radial intensity profile
        gauss_peaks = find_all_peaks(radial_intensity, window_size=3)

        # For each q-peak reference value, calculate the sample-to-detector
        # distance
        q_peaks_ref = self.q_peaks_ref


        import matplotlib.pyplot as plt
        fig1 = plt.figure()
        # Plot original image
        plt.imshow(image)

        # Plot centerized image
        fig2 = plt.figure()
        plt.imshow(centered_image)

        # Plot polar image
        fig3 = plt.figure()
        plt.imshow(polar_image)

        # Plot intensity vs. pixel radius
        fig4 = plt.figure()
        plt.scatter(r_space, radial_intensity)

        plt.show()

        return distance


    def sample_set_detector_distance(self):
        """
        Calculate the sample-to-detector distance for an entire sample set
        """
        pass
