"""
Store calibration materials data

q units are per meter
"""

# AgBH q-peaks via:
# http://gisaxs.com/index.php/Material:Silver_behenate 

q_peaks_ref_dict_m = {
        "silver_behenate": {
            "singlets": [
                0.1076e10,
                0.2152e10,
                0.3228e10,
                0.4304e10,
                0.5380e10,
                0.6456e10,
                0.7532e10,
                0.8608e10,
                0.9684e10,
                1.076e10,
                1.184e10,
                ],
            "doublets": [
                1.369e10,
                1.387e10,
                ],
        }
}
