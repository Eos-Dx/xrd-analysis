# hardware/Ulster/gui/main_window_ext/drawing_extension.py
from PyQt5.QtWidgets import QAction, QActionGroup


class DrawingMixin:

    def createDrawingActions(self):
        # Create actions for drawing modes.
        self.selectRectAct = QAction("Rectangle", self, checkable=True, triggered=self.selectRectMode)
        self.selectEllipseAct = QAction("Circle", self, checkable=True, triggered=self.selectEllipseMode)
        self.cropAct = QAction("Crop", self, checkable=True, triggered=self.selectCropMode)
        self.selectAct = QAction("Select", self, checkable=True, triggered=self.selectSelectMode)

        self.drawingModeGroup = QActionGroup(self)
        self.drawingModeGroup.addAction(self.selectRectAct)
        self.drawingModeGroup.addAction(self.selectEllipseAct)
        self.drawingModeGroup.addAction(self.cropAct)
        self.drawingModeGroup.addAction(self.selectAct)

        # Default to "Select" mode.
        self.selectAct.setChecked(True)

    def addDrawingActionsToToolBar(self):
        # Assumes self.toolBar exists (created in MainWindowBasic).
        self.toolBar.addAction(self.selectRectAct)
        self.toolBar.addAction(self.selectEllipseAct)
        self.toolBar.addAction(self.cropAct)
        self.toolBar.addAction(self.selectAct)

    # Methods that change the drawing mode of the image view:
    def selectRectMode(self):
        self.image_view.setDrawingMode("rect")

    def selectEllipseMode(self):
        self.image_view.setDrawingMode("ellipse")

    def selectCropMode(self):
        self.image_view.setDrawingMode("crop")

    def selectSelectMode(self):
        self.image_view.setDrawingMode(None)
