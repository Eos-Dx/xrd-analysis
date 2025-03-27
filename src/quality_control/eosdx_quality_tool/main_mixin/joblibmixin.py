import joblib
from quality_control.eosdx_quality_tool.utility.joblib_handler import compute_statistics
from quality_control.eosdx_quality_tool.utility.data_pipeline import process_dataframe


class JoblibHandlerMixin:
    def load_and_process_joblib_file(self, file_path):
        """
        Loads a joblib file (expected to contain a DataFrame), processes it using the XRD pipeline,
        computes statistics on the original DataFrame, and returns a tuple:
        (original_df, original_stats, transformed_df).
        """
        try:
            original_df = joblib.load(file_path)
            original_stats = compute_statistics(original_df)
            transformed_df = process_dataframe(original_df)
            return original_df, original_stats, transformed_df
        except Exception as e:
            raise Exception(f"Error loading or processing file: {e}")
