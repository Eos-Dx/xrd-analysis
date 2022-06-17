"""
Calculate diffraction pattern properties
"""
import numpy as np

PIXEL_WIDTH = 55e-6 # 55 um
WAVELENGTH = 1.5418e-10 # 1.5418 Angstrom
HELIX_PITCH = 5.1e-10 # 5.1 Angstrom
INTERAXIAL_DISTANCE = 9.8e-10 # 9.8 Angstrom

theta_helix_pitch = np.arcsin(WAVELENGTH/HELIX_PITCH)
sample_detector_shift_for_single_pixel_shift_of_helix_pitch_feature = PIXEL_WIDTH/np.tan(theta_helix_pitch)
helix_pitch_feature_shift_for_1mm_sample_detector_shift = 1e-3*np.tan(theta_helix_pitch)/PIXEL_WIDTH

print("Sample-detector distance shift required to shift the 5.1 A helix pitch feature by 1 pixel (mm):")
print(sample_detector_shift_for_single_pixel_shift_of_helix_pitch_feature*1000, "mm")

print("Sample-detector distance shift of 1 mm would shift the 5.1 A helix pitch feature by:")
print(helix_pitch_feature_shift_for_1mm_sample_detector_shift, "pixels")
