Data Validation
=========================================

Abstract
--------
This document describes the types of validation checks performed to ensure data quality during X-ray diffraction (XRD) measurements. These checks align with standard XRD practices to ensure accuracy and reliability of the data.

There are three types of validation:

1. Form validation
2. Measurement validation
3. Manual validation


Form Validation
---------------
Form validations are performed by restricting inputs to predefined values to maintain consistency in data entry. The following parameters are validated through the form:

* **Existence of files:** Confirms all required files, such as measurement data and standards, are uploaded before analysis.
* **Project:** Ensures that the project is correctly associated with the data.
* **Machine:** Verifies that the selected machine is correct for the chosen project.
* **Date:** Confirms that the entered date is valid.
* **Measurement distance:** Ensures the measurement distance is within acceptable bounds based on instrument configuration.
* **Poni file:** Ensures that a calibration file is present to correct for detector geometry. Missing or incorrect calibration files will lead to inaccurate diffraction patterns.
* **Metadata:** Verifies that proper metadata (e.g., patient ID, sample ID, and experiment details) is provided for traceability.

Measurement Validation
-----------------------
Measurement validation focuses on ensuring the reliability and accuracy of the XRD data collected. Some checks are specific to particular types of measurements, while others are general and apply to most XRD data.

Shared Validation Checks
*************************

These checks apply to all types of XRD measurements (calibration, tissue, etc.):

* **Symmetry of diffraction pattern:** The diffraction pattern should be radially symmetric around the beam center. Asymmetry may indicate issues with sample alignment or equipment calibration.
* **Center shape check:** The center of the diffraction pattern should be circular. Deformed or irregular shapes could indicate misalignment or detector issues.
* **Center position check:** Confirms that the beam center is correctly aligned. Misalignment leads to distorted diffraction data and could invalidate measurements.
* **Beam peak position:** Diffraction patterns should exhibit decaying intensity with the highest value near zero. Deviations from this expected behavior can indicate instrument or calibration issues.
* **Overall intensity check:** Ensures that the diffraction pattern intensity remains stable across repeated measurements. Variations may indicate sample preparation errors, beam inconsistencies, or detector sensitivity problems.

Calibration Validation Checks
*****************************

Calibration measurements require additional checks to verify the accuracy of the instrument calibration:

* **Poni file consistency:** Verifies that the **Poni file** used in calibration matches the manually inputted distance within acceptable tolerance. A mismatch can indicate calibration errors.
* **Peak consistency (±1%):** The positions of the diffraction peaks in well-studied calibration materials should remain consistent across repeated measurements. Deviations greater than 1% may point to issues like instrument drift, temperature fluctuation, or sample changes.

Tissue Measurements Validation Checks
*************************************

Tissue samples present unique challenges compared to crystalline standards. These checks are designed specifically for biological samples:

* **Background signal check:** Tissue measurements often have weaker signals. This check ensures that the signal-to-noise ratio is sufficient for reliable analysis. Excess noise could indicate contamination, poor sample quality, or improper instrument settings.


Additional Measurement Types
----------------------------
The following checks might be relevant for additional measurement types in the future, such as background, empty, and dark measurements:

* **Background measurements:** Ensures no significant diffraction features are present in the background, indicating that any observed features come from the sample and not environmental noise or contamination.

* **Empty measurements:** Confirms that no diffraction pattern is present when no sample is loaded, ensuring that the instrument is correctly calibrated.

* **Dark measurements:** Verifies that no signal is detected when the X-ray beam is off, ensuring the detector is functioning properly and there is no electronic noise affecting the measurements.

Manual Validation
-----------------
Manual validation is performed by the quality control (QC) team after all automated checks are completed. The team visually inspects the diffraction patterns, checks logs for inconsistencies, and compares results with historical data to ensure the measurement meets the expected quality standards.

Tools
-----
To perform the various validation checks outlined in this document, several tools and libraries will be integrated into the workflow. These tools ensure both automated and manual validations can be carried out efficiently and accurately:

#. **pyFAI:** A Python library for azimuthal integration of diffraction images. pyFAI will be used for:
    * **1D and 2D azimuthal integration** to transform diffraction images into usable data (radial profiles).
    * **Calibration checks**, such as verifying the consistency of the Poni file with manual distances and performing symmetry checks.
    * **Intensity checks** by enabling integration of diffraction data to compare intensities across repeated measurements.

#. **SciPy (for peak analysis):** SciPy provides functions for detecting peaks in 1D data, which will be critical for:
    * **Main peak position checks** to ensure the highest value is around 0 for both calibration and tissue measurements.
    * **Peak consistency checks** by comparing peak positions across repeated measurements.

#. **NumPy (for data consistency):** NumPy will be employed for general numerical operations required in checks like:
    * **Intensity comparison** across different measurements.
    * **Background noise subtraction** to ensure proper signal-to-noise ratio in tissue samples.

#. **Matplotlib:** A plotting library used to generate visual representations of diffraction data for:
    * **Visual inspection** of diffraction patterns during manual validation.
    * **Graphing radial profiles** to easily identify issues with intensity and symmetry.

#. **h5py (for HDF5 integration):** As diffraction data is stored in HDF5 format, h5py will be used for:
    * **Managing and retrieving patient and sample-specific metadata** associated with diffraction measurements.
    * **Validating the existence and consistency of calibration files, measurement data, and associated metadata** in the HDF5 structure.

#. *UNDER CONSIDERATION* **xrayutilities (for advanced XRD analysis):** A Python library designed for X-ray diffraction and scattering data analysis. It will help with:
    * **Advanced peak fitting and lattice parameter determination**, which could be useful for more detailed tissue and calibration measurements.
    * **Angle and position checks** of the diffraction peaks.
