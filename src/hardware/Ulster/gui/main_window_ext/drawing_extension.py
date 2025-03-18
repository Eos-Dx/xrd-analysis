from PyQt5.QtWidgets import QAction, QActionGroup
from PyQt5.QtCore import QRectF
from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsItem


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
        # Add drawing actions to the existing toolbar.
        self.toolBar.addAction(self.selectRectAct)
        self.toolBar.addAction(self.selectEllipseAct)
        self.toolBar.addAction(self.cropAct)
        self.toolBar.addAction(self.selectAct)

    # Mode switching methods:
    def selectRectMode(self):
        self.image_view.setDrawingMode("rect")

    def selectEllipseMode(self):
        self.image_view.setDrawingMode("ellipse")

    def selectCropMode(self):
        self.image_view.setDrawingMode("crop")

    def selectSelectMode(self):
        self.image_view.setDrawingMode(None)

    # Delete action:
    def createDeleteAction(self):
        self.deleteAct = QAction("Delete", self, triggered=self.deleteSelectedShapes)

    def addDeleteActionToToolBar(self):
        self.toolBar.addAction(self.deleteAct)

    def deleteSelectedShapes(self):
        if hasattr(self, 'image_view'):
            self.image_view.deleteSelectedShapes()
