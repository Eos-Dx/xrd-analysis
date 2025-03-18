from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QSpinBox, QPushButton, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QBrush, QPen
from PyQt5.QtWidgets import QGraphicsEllipseItem


class ZonePointsMixin:
    def createZonePointsWidget(self):
        """Creates a dock widget that allows generating points within the include zone,
        and deleting selected points from the generated set."""
        self.zonePointsDock = QDockWidget("Zone Points", self)
        container = QWidget()
        layout = QVBoxLayout(container)

        # Input controls for grid resolution and generate button.
        inputLayout = QHBoxLayout()
        self.gridSpinBox = QSpinBox()
        self.gridSpinBox.setMinimum(2)
        self.gridSpinBox.setMaximum(1000)
        self.gridSpinBox.setValue(10)  # Default: 10 points per row/column.
        inputLayout.addWidget(self.gridSpinBox)

        self.generatePointsBtn = QPushButton("Generate Points")
        inputLayout.addWidget(self.generatePointsBtn)
        layout.addLayout(inputLayout)

        # Table to show generated points.
        self.pointsTable = QTableWidget(0, 4)
        self.pointsTable.setHorizontalHeaderLabels(["ID", "X", "Y", "Measurement"])
        layout.addWidget(self.pointsTable)

        # Delete button for selected points.
        self.deletePointsBtn = QPushButton("Delete Selected Points")
        layout.addWidget(self.deletePointsBtn)

        container.setLayout(layout)
        self.zonePointsDock.setWidget(container)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.zonePointsDock)

        self.generatePointsBtn.clicked.connect(self.generateZonePoints)
        self.deletePointsBtn.clicked.connect(self.deleteSelectedPoints)

    def generateZonePoints(self):
        """Generate a grid of points inside the include zone while avoiding exclude zones."""
        # Find the include shape and exclude shapes.
        include_shape = None
        exclude_shapes = []
        for shape in self.image_view.shapes:
            role = shape.get("role", "include")
            if role == "include":
                include_shape = shape["item"]
            elif role == "exclude":
                exclude_shapes.append(shape["item"])
        if include_shape is None:
            print("No include shape defined. Cannot generate points.")
            return

        # Clear previously generated points.
        if hasattr(self.image_view, 'generated_points'):
            for pt_item in self.image_view.generated_points:
                self.image_view.scene.removeItem(pt_item)
            self.image_view.generated_points = []
        else:
            self.image_view.generated_points = []

        # Use the include shape's bounding rectangle for sampling.
        include_rect = include_shape.rect() if hasattr(include_shape, "rect") else include_shape.boundingRect()
        n = self.gridSpinBox.value()
        dx = include_rect.width() / (n - 1) if n > 1 else include_rect.width()
        dy = include_rect.height() / (n - 1) if n > 1 else include_rect.height()

        points = []
        for i in range(n):
            for j in range(n):
                x = include_rect.x() + i * dx
                y = include_rect.y() + j * dy
                # Check if point is strictly inside the include shape.
                if not include_shape.contains(include_shape.mapFromScene(x, y)):
                    continue
                # Skip point if it falls within any exclude zone.
                skip = False
                for ex in exclude_shapes:
                    if ex.contains(ex.mapFromScene(x, y)):
                        skip = True
                        break
                if not skip:
                    points.append((x, y))

        # Draw the points and store them.
        for (x, y) in points:
            pt_item = QGraphicsEllipseItem(x - 2, y - 2, 4, 4)
            pt_item.setBrush(QColor("blue"))
            pt_item.setPen(QPen(Qt.NoPen))
            # Make the point selectable and movable.
            pt_item.setFlags(QGraphicsEllipseItem.ItemIsSelectable | QGraphicsEllipseItem.ItemIsMovable)
            self.image_view.scene.addItem(pt_item)
            self.image_view.generated_points.append(pt_item)

        self.updatePointsTable(points)

    def updatePointsTable(self, points):
        self.pointsTable.setRowCount(len(points))
        for idx, (x, y) in enumerate(points):
            self.pointsTable.setItem(idx, 0, QTableWidgetItem(str(idx + 1)))
            self.pointsTable.setItem(idx, 1, QTableWidgetItem(f"{x:.2f}"))
            self.pointsTable.setItem(idx, 2, QTableWidgetItem(f"{y:.2f}"))
            self.pointsTable.setItem(idx, 3, QTableWidgetItem(""))  # Placeholder.

    def deleteSelectedPoints(self):
        """Deletes points selected in the points table and removes them from the scene."""
        selected_indexes = self.pointsTable.selectionModel().selectedRows()
        rows = [index.row() for index in selected_indexes]
        rows.sort(reverse=True)
        for row in rows:
            try:
                if hasattr(self.image_view, 'generated_points') and row < len(self.image_view.generated_points):
                    pt_item = self.image_view.generated_points[row]
                    self.image_view.scene.removeItem(pt_item)
                    del self.image_view.generated_points[row]
            except Exception as e:
                print("Error deleting point:", e)
        # Rebuild points list from remaining items.
        new_points = []
        for pt in self.image_view.generated_points:
            rect = pt.rect()
            x = rect.x() + rect.width() / 2
            y = rect.y() + rect.height() / 2
            new_points.append((x, y))
        self.updatePointsTable(new_points)
