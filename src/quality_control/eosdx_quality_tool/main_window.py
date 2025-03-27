import quality_control.eosdx_quality_tool.config as config
from PyQt5.QtWidgets import QMainWindow, QAction
from quality_control.eosdx_quality_tool.main_mixin.filemixin import FileDialogMixin
from quality_control.eosdx_quality_tool.main_mixin.joblibmixin import JoblibHandlerMixin
from quality_control.eosdx_quality_tool.main_mixin.dataframestatsmixin import DataFrameStatsMixin
from quality_control.eosdx_quality_tool.main_mixin.visualizationmixin import VisualizationMixin


class MainWindow(QMainWindow, FileDialogMixin, JoblibHandlerMixin, DataFrameStatsMixin, VisualizationMixin):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EosDX Quality Tool")
        self.resize(1200, 800)

        # Initialize the DataFrame statistics zone (dock widget)
        self.init_df_stats_zone()
        # Initialize the visualization zone (central widget)
        self.init_visualization_zone()

        # Variable to store the transformed DataFrame
        self.transformed_df = None

        self.initMenu()

        # If in development mode, load the default joblib file
        if config.DEV:
            self.load_default_joblib()

    def initMenu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Joblib", self)
        open_action.triggered.connect(self.open_joblib_file)
        file_menu.addAction(open_action)

    def open_joblib_file(self):
        file_path = self.open_file_dialog("Joblib Files (*.joblib);;All Files (*)")
        if file_path:
            try:
                # Load the joblib file; get original dataframe, its statistics, and the transformed dataframe.
                original_df, original_stats, transformed_df = self.load_and_process_joblib_file(file_path)
                self.transformed_df = transformed_df

                # Update the DataFrame statistics zone (dock widget) with original stats.
                self.update_df_stats(f"Original DataFrame Statistics:\n{original_stats}")
                # Update the visualization zone with the first measurement.
                self.display_measurement(0)
            except Exception as e:
                self.update_df_stats(str(e))

    def load_default_joblib(self):
        default_path = config.DEFAULT_JOBLIB
        try:
            original_df, original_stats, transformed_df = self.load_and_process_joblib_file(default_path)
            self.transformed_df = transformed_df
            self.update_df_stats(f"Default Original DataFrame Statistics:\n{original_stats}")
            self.display_measurement(0)
        except Exception as e:
            self.update_df_stats(str(e))
