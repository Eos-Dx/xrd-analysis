"""
Calibration materials reference data
"""

# AgBH q-peaks via:
# http://gisaxs.com/index.php/Material:Silver_behenate 
# D-spacing is 5.8 nm
# q-peak units are inverse Angstroms

q_peaks_ref_dict = {
        "silver_behenate": {
            "singlets": [
                0.1076,
                0.2152,
                0.3228,
                0.4304,
                0.5380,
                0.6456,
                0.7532,
                0.8608,
                0.9684,
                1.076,
                1.184,
                ],
            "doublets": [
                1.369,
                1.387,
                ],
        }
}

CALIBRATION_MATERIAL_LIST = q_peaks_ref_dict.keys()
