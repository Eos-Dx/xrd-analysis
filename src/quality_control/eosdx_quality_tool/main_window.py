import quality_control.eosdx_quality_tool.config as config
from pathlib import Path
from PyQt5.QtWidgets import QMainWindow, QAction
from PyQt5.QtGui import QIcon
from quality_control.eosdx_quality_tool.main_mixin.filemixin import FileDialogMixin
from quality_control.eosdx_quality_tool.main_mixin.h5mixin import H5HandlerMixin
from quality_control.eosdx_quality_tool.main_mixin.dataframestatsmixin import DataFrameStatsMixin
from quality_control.eosdx_quality_tool.main_mixin.visualizationmixin import VisualizationMixin
from quality_control.eosdx_quality_tool.main_mixin.exclude_include_mixin import ExcludeIncludeMixin
from quality_control.eosdx_quality_tool.main_mixin.excluded_included_filesmixin import ExcludedIncludedFilesMixin


class MainWindow(QMainWindow, FileDialogMixin, H5HandlerMixin, DataFrameStatsMixin, VisualizationMixin,
                 ExcludeIncludeMixin, ExcludedIncludedFilesMixin):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EosDX Quality Tool")
        self.resize(700, 500)
        logo_path = Path('logo_quality.png')  # Adjust the path.
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))
        else:
            print("Logo file not found:", logo_path)
        # Initialize the DataFrame statistics zone (dock widget)
        self.init_df_stats_zone()
        # Initialize the visualization zone (central widget)
        self.init_visualization_zone()
        # Initialize the exclude zone (dock widget)
        self.init_exclude_zone()


        # Variables to store data from the HDF5 file
        self.calibration_df = None
        self.measurement_df = None
        self.transformed_df = None

        self.initMenu()

        # In development mode, automatically load the default HDF5 file.
        if config.DEV:
            self.load_default_h5()

        # Initialize the excluded files zone (dock widget)
        self.init_excluded_included_files_zone()

    def initMenu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open H5", self)
        open_action.triggered.connect(self.open_h5_file_dialog)
        file_menu.addAction(open_action)

    def open_h5_file(self, file_path):
        file_path = Path(file_path)
        if file_path.exists():
            try:
                # Convert the HDF5 file into calibration and measurement DataFrames,
                # compute statistics on measurement_df, and process it through the pipeline.
                calibration_df, measurement_df, measurement_stats, transformed_df = self.load_and_process_h5_file(
                    file_path)
                self.file_path = file_path
                self.calibration_df = calibration_df
                self.measurement_df = measurement_df
                self.transformed_df = transformed_df
                self.update_df_stats(f"Measurement DataFrame Statistics:\n{measurement_stats}")
                self.display_measurement(0)
                self.update_measurements_list()
            except Exception as e:
                self.update_df_stats(str(e))

    def open_h5_file_dialog(self):
        file_path = Path(self.open_file_dialog("HDF5 Files (*.h5 *.hdf5);;All Files (*)"))
        self.open_h5_file(file_path)

    def load_default_h5(self):
        default_path = config.DEFAULT_H5  # Make sure your config has DEFAULT_H5 defined.
        self.open_h5_file(default_path)
