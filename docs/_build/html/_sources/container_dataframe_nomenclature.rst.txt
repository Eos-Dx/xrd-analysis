==============================================
Container -> DataFrame Nomenclature
==============================================

This document provides a description of how containers are represented as DataFrames in the context of X-ray Diffraction (XRD) data analysis.

.. contents:: Table of Contents
   :local:
   :depth: 2

Introduction
--------------

This documentation outlines the mapping between container metadata and DataFrame column names in the XRD analysis pipeline. The tables below define how original container fields are transformed and represented in the resulting DataFrames, supporting efficient data processing and analysis.

Calibration and Machine Configuration
---------------------------------------

This table details the metadata related to instrument calibration and machine configuration settings for XRD measurements, organized by functional categories.

Identification and Core Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Container Metadata Name
     - DataFrame Column Name
     - Description
   * - Dataset
     - measurement_data
     - Two-dimensional matrix containing the raw diffraction image data.
   * - Dataset Name
     - cal_name
     - Identifier for the source raw data file from which measurements were extracted.
   * - Group name
     - id/calib_name
     - Cross-reference key linking sample measurements to their corresponding calibration data; represents the group name in H5 files that contains both calibrations and measurements.
   * - metadata
     - calib_metadata
     - Raw metadata associated with the measurement file, typically stored in DSC file format or natively within H5/GFRM containers.

Instrument Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Container Metadata Name
     - DataFrame Column Name
     - Description
   * - source
     - calib_source
     - X-ray source material (Cu, Mo, Ag, etc.) which determines the characteristic wavelength used in the experiment.
   * - wavelength
     - calib_wavelength
     - X-ray wavelength used for data collection, measured in Angstroms (Å).
   * - distanceInMM
     - calib_distanceInMM
     - Physical distance between the sample and detector measured in millimeters, critical for accurate 2D-to-1D data conversion.
   * - pixelSize
     - calib_pixelSize
     - Physical size of individual detector pixels measured in micrometers, essential for spatial calibration.
   * - matrixResolution
     - calib_matrixResolution
     - Resolution dimensions of the detector matrix used for measurement, typically expressed as width × height in pixels.
   * - machineName
     - calib_machineName
     - Identifier for the specific diffractometer or instrument used for data collection as recorded in the database.

Experimental Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Container Metadata Name
     - DataFrame Column Name
     - Description
   * - exposure
     - calib_exposure
     - Duration of X-ray exposure in seconds. This is the preferred exposure time field.
   * - exposureTime *(legacy)*
     - calib_exposureTime *(legacy)*
     - Legacy field for duration of X-ray exposure in seconds. Use calib_exposure when available.
   * - measurement_timestamp
     - calib_measurement_timestamp
     - Timestamp recording when the measurement was collected, stored as datetime format. This is the preferred and more reliable timestamp field.
   * - timestamp *(legacy)*
     - calib_timestamp *(legacy)*
     - Legacy timestamp field recording when the measurement was collected, stored as UNIX time. Use calib_measurement_timestamp when available.

Study Information
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Container Metadata Name
     - DataFrame Column Name
     - Description
   * - studyName
     - calib_studyName
     - Name of the research study with which these measurements are associated, as recorded in the database.

Advanced Processing
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Container Metadata Name
     - DataFrame Column Name
     - Description
   * - ponifile
     - ponifile
     - Reference to the parameter file used by pyFAI for azimuthal integration, containing beam center coordinates and precise distance calibrations.

Sample Measurements
---------------------

This table details the metadata related to sample information and measurement parameters for XRD analysis.

Measurement Core Data
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Container Metadata Name
     - DataFrame Column Name
     - Description
   * - Dataset
     - measurement_data
     - Two-dimensional matrix containing the raw diffraction image data.
   * - Dataset Name
     - meas_name
     - Identifier for the source raw data file from which measurements were extracted.
   * - Group name
     - id
     - Cross-reference key linking sample measurements to their corresponding calibration data; represents the group name in H5 files that contains both calibrations and measurements.
   * - metadata
     - metadata
     - Raw metadata associated with the measurement file, typically stored in DSC file format or natively within H5/GFRM containers.
   * - measurement_timestamp
     - measurement_timestamp
     - Precise datetime when the measurement was recorded.
   * - exposure
     - exposure
     - Duration of X-ray exposure in seconds for this specific sample measurement.
   * - measurementsGroupId
     - measurementsGroupId
     - Database identifier linking related measurements performed as part of a single experimental group.

Specimen Information
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Container Metadata Name
     - DataFrame Column Name
     - Description
   * - specimenId
     - specimenId
     - Laboratory-assigned identifier for the biological specimen or sample.
   * - specimenDBId
     - specimenDBId
     - Unique database identifier for the specimen record associated with this measurement.
   * - specimenType
     - specimenType
     - Anatomical origin of the specimen (e.g., breast, prostate, skin).
   * - specimenStatus
     - specimenStatus
     - Pathological classification of the specimen (e.g., cancerous, normal, benign).
   * - species
     - species
     - Biological species of the specimen (human, mouse, etc.).
   * - organ
     - organ
     - Specific tissue or organ from which the sample was collected (e.g., skin, claw, etc.).
   * - organSide
     - organSide
     - Anatomical side designation for paired organs (e.g., Left/Right breast).
   * - hoursSinceInoculation
     - hoursSinceInoculation
     - Time elapsed in hours between specimen inoculation and collection (primarily for research specimens).

Patient Information
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Container Metadata Name
     - DataFrame Column Name
     - Description
   * - patientId
     - patientId
     - Laboratory-assigned identifier for the patient from whom the specimen was collected.
   * - patientDBId
     - patientDBId
     - Unique database identifier for the patient record associated with this measurement.
   * - age
     - age
     - Patient's age in years at the time of specimen collection.
   * - cohort
     - cohort
     - Research or study group to which the sample belongs.

Cancer Classification
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Container Metadata Name
     - DataFrame Column Name
     - Description
   * - isCancerDiagnosed
     - isCancerDiagnosed
     - Boolean indicator of whether the patient has received a cancer diagnosis.
   * - grade
     - grade
     - Clinical assessment of cancer progression level or severity.
   * - cancerCategoryName
     - cancerCategoryName
     - Classification of cancer category according to World Health Organization (WHO) standards.
   * - cancerFamilyName
     - cancerFamilyName
     - Classification of cancer family according to World Health Organization (WHO) taxonomy.
   * - cancerTypeName
     - cancerTypeName
     - Specific cancer type classification according to World Health Organization (WHO) guidelines.
   * - cancerSubtype
     - cancerSubtype
     - Further detailed classification of the specific cancer subtype.
   * - cancerMixed
     - cancerMixed
     - Boolean indicator denoting whether the cancer consists of multiple types.
