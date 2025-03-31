from xrdanalysis.data_processing.utility_functions import h5_to_df
from quality_control.eosdx_quality_tool.utility.joblib_handler import compute_statistics
from quality_control.eosdx_quality_tool.utility.data_pipeline import process_dataframe, process_dataframe_2D


class H5HandlerMixin:
    def load_and_process_h5_file(self, file_path):
        """
        Loads an HDF5 file, converts it into calibration and measurement DataFrames using h5_to_df,
        computes statistics on the measurement DataFrame, and processes the measurement DataFrame
        using the XRD pipeline.

        :param file_path: Path to the HDF5 file.
        :type file_path: str
        :return: A tuple containing:
                 - calibration_df (pandas.DataFrame)
                 - measurement_df (pandas.DataFrame)
                 - measurement_stats (str) computed from measurement_df
                 - transformed_df (pandas.DataFrame) resulting from processing measurement_df
        :rtype: Tuple[pd.DataFrame, pd.DataFrame, str, pd.DataFrame]
        """
        # Convert the HDF5 file into two DataFrames
        calibration_df, measurement_df = h5_to_df(file_path)

        # Compute statistics on the measurement DataFrame
        measurement_stats = compute_statistics(measurement_df)

        # Process the measurement DataFrame using the XRD pipeline
        transformed_df = process_dataframe(measurement_df)
        #calibration_df = process_dataframe_2D(calibration_df)

        return calibration_df, measurement_df, measurement_stats, transformed_df
