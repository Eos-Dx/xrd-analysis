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

    def __init__(self, calibration_material):
        """
        Initialize Calibration class
        """
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
        polar_image = warp_polar(centered_image)

        # Convert to 1D intensity vs. radius (pixel)
        final_r = int(r_max) if r_max else None
        radial_intensity = np.sum(polar_image, axis=0)[:final_r]

        import matplotlib.pyplot as plt
        plt.scatter(range(len(radial_intensity)), radial_intensity)
        plt.show()

        # For each q-peak reference value, calculate
        # the sample-to-detector distance


    def sample_set_detector_distance(self):
        """
        Calculate the sample-to-detector distance for an entire sample set
        """
        pass
