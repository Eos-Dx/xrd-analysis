from PyQt5.QtWidgets import QFileDialog


class FileDialogMixin:
    def open_file_dialog(self, file_filter="All Files (*)"):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", file_filter, options=options)
        return file_path

