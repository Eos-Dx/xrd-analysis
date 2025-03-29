from PyQt5.QtWidgets import QDockWidget, QTextEdit
from PyQt5.QtCore import Qt

class DataFrameStatsMixin:
    def init_df_stats_zone(self):
        """
        Initializes a dockable widget to display DataFrame statistics.
        Currently uses a QTextEdit but can be extended later.
        """
        self.df_stats_dock = QDockWidget("DataFrame Statistics", self)
        self.df_stats_text = QTextEdit()
        self.df_stats_text.setReadOnly(True)
        self.df_stats_dock.setWidget(self.df_stats_text)
        # Add the dock widget to the left area of the main window
        self.addDockWidget(Qt.LeftDockWidgetArea, self.df_stats_dock)

    def update_df_stats(self, text):
        """
        Updates the DataFrame statistics zone with the provided text.
        """
        self.df_stats_text.setText(text)
