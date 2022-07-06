"""
Code to calibrate X-ray diffraction setup using calibration samples data
"""

from eosdxanalysis.calibration.materials import q_peaks_ref_dict


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

    def single_sample_detector_distance(self):
        """
        Calculate the sample-to-detector distance for a single sample
        """
        pass

    def sample_set_detector_distance(self):
        """
        Calculate the sample-to-detector distance for an entire sample set
        """
        pass
