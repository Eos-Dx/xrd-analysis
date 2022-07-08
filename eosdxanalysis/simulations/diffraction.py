"""
Calculate diffraction pattern properties

The helix pitch yields a first-order peak.

Using Bragg's equation:
    n*wavelength = 2*d*sin(theta)

Set n=1 to get the first-order peak of the alpha helix pitch.

Solving for theta gives:
    theta = arcsin(wavelength/(2*d))

Let
- z be the beam direction,
- y the meridional direction in the diffraction image, and
- x the equatorial direction in the diffraction image.

"""
import numpy as np

PIXEL_WIDTH = 55e-6 # 55 um
WAVELENGTH = 1.5418e-10 # 1.5418 Angstrom
HELIX_PITCH = 5.1e-10 # 5.1 Angstrom
INTERAXIAL_DISTANCE = 9.8e-10 # 9.8 Angstrom

# Using Bragg's equation for the first-order peak, solving for theta yields
alpha_helix_theta = np.arcsin(WAVELENGTH/HELIX_PITCH/2)

# Now that we have the angle, we use tan(theta) = y-shift/z-shift, and
# solve for the z-shift that corresponds to a y-shift of one pixel width
dy = PIXEL_WIDTH
sample_detector_shift = dy/np.tan(alpha_helix_theta)

# Similarly, we solve for the peak shift that corresponds to a sample-to-detector
# distance shift of 1 mm
dz = 1e-3 # 1e-3 meters
alpha_helix_peak_shift_meters = dz*np.tan(alpha_helix_theta)
alpha_helix_peak_shift_um = alpha_helix_peak_shift_meters * 1e6
# Convert y-shift to pixels
alpha_helix_peak_shift_pixels = alpha_helix_peak_shift_meters/PIXEL_WIDTH

print("Sample-detector distance shift required to shift the "
        "5.1 A alpha helix first-order peak by 1 pixel:")
print(np.round(sample_detector_shift*1000, decimals=2), "mm")

print("Sample-detector distance shift of 1 mm would shift the "
        "5.1 A helix pitch feature by:")
print(np.round(alpha_helix_peak_shift_um), "um")
print("or")
print(np.round(alpha_helix_peak_shift_pixels, decimals=2), "pixels")
