from PyQt5.QtWidgets import QToolButton, QMenu, QAction
from PyQt5.QtCore import Qt


class RotatorToolButton(QToolButton):
    def __init__(self, text, defaultAngle, rotateCallback, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.defaultAngle = defaultAngle
        self.rotateCallback = rotateCallback
        self.menu = QMenu(self)
        for angle in [0.5, 1, 2, 5, 10]:
            action = QAction(f"{angle}°", self)
            action.setData(angle)
            self.menu.addAction(action)
        self.menu.triggered.connect(self.onMenuTriggered)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.menu.exec_(event.globalPos())
        else:
            super().mousePressEvent(event)

    def onMenuTriggered(self, action):
        angle = action.data()
        self.defaultAngle = angle

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.rotateCallback(self.defaultAngle)
        super().mouseReleaseEvent(event)


class RotationMixin:
    def createRotationActions(self):
        # Create rotate left and rotate right buttons.
        self.rotateLeftBtn = RotatorToolButton("Rotate Left", 1, self.rotateLeft, self)
        self.rotateRightBtn = RotatorToolButton("Rotate Right", 1, self.rotateRight, self)

    def addRotationActionsToToolBar(self):
        self.toolBar.addWidget(self.rotateLeftBtn)
        self.toolBar.addWidget(self.rotateRightBtn)

    def rotateLeft(self, angle):
        if hasattr(self.image_view, 'rotateImage'):
            self.image_view.rotateImage(-angle)

    def rotateRight(self, angle):
        if hasattr(self.image_view, 'rotateImage'):
            self.image_view.rotateImage(angle)
