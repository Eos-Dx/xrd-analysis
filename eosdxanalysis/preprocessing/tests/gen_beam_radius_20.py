"""
Generates a synthetic diffraction image with beam radius 20
"""

from eosdxanalysis.preprocessing.beam_utils import azimuthal_integration

import numpy as np

beam_radius = 2
beam_amplitude = 10
peak_location = 2
peak_amplitude = 1

x_start = -10
y_start = x_start
x_end = 10
y_end = x_end

shape = (256,256)
YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]

RR = np.sqrt(XX**2 + YY**2)

beam_function = beam_radius - RR
beam_function[RR > 2] = 0
peak_function = peak_amplitude*np.exp(-(RR - peak_location)**2)
pattern_function = beam_function + peak_function

profile_1d_pattern = azimuthal_integration(pattern_function)
profile_1d_beam_function = azimuthal_integration(beam_function)
profile_1d_peak_function = azimuthal_integration(peak_function)
