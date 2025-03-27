from PyQt5.QtWidgets import QGraphicsEllipseItem


class HoverableEllipseItem(QGraphicsEllipseItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptHoverEvents(True)
        self.hoverCallback = None

    def hoverEnterEvent(self, event):
        if self.hoverCallback:
            self.hoverCallback(self, True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        if self.hoverCallback:
            self.hoverCallback(self, False)
        super().hoverLeaveEvent(event)
