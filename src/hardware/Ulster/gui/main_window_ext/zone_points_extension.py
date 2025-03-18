from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QSpinBox, QPushButton, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QBrush, QPen
from PyQt5.QtWidgets import QGraphicsEllipseItem


class ZonePointsMixin:
    def createZonePointsWidget(self):
        """Creates a dock widget that allows the user to generate a grid of points
        within the 'include' zone (avoiding any 'exclude' zones)."""
        self.zonePointsDock = QDockWidget("Zone Points", self)
        container = QWidget()
        layout = QVBoxLayout(container)

        # Input controls for grid resolution.
        inputLayout = QHBoxLayout()
        self.gridSpinBox = QSpinBox()
        self.gridSpinBox.setMinimum(2)
        self.gridSpinBox.setMaximum(1000)
        self.gridSpinBox.setValue(10)  # default: 10 points per row/column.
        inputLayout.addWidget(self.gridSpinBox)

        self.generatePointsBtn = QPushButton("Generate Points")
        inputLayout.addWidget(self.generatePointsBtn)
        layout.addLayout(inputLayout)

        # Table to show generated points.
        self.pointsTable = QTableWidget(0, 4)
        self.pointsTable.setHorizontalHeaderLabels(["ID", "X", "Y", "Measurement"])
        layout.addWidget(self.pointsTable)

        container.setLayout(layout)
        self.zonePointsDock.setWidget(container)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.zonePointsDock)

        self.generatePointsBtn.clicked.connect(self.generateZonePoints)

    def generateZonePoints(self):
        """Generate a grid of points inside the include shape while avoiding exclude zones.
           Assumes there is exactly one include shape and zero or more exclude shapes.
           The grid resolution is defined by gridSpinBox (points per row/column)."""
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

        # Get the bounding rectangle of the include shape (for grid sampling).
        include_rect = include_shape.rect() if hasattr(include_shape, "rect") else include_shape.boundingRect()
        n = self.gridSpinBox.value()

        # Compute step sizes along x and y.
        dx = include_rect.width() / (n - 1) if n > 1 else include_rect.width()
        dy = include_rect.height() / (n - 1) if n > 1 else include_rect.height()

        points = []
        # Loop over grid positions in the bounding rectangle.
        for i in range(n):
            for j in range(n):
                x = include_rect.x() + i * dx
                y = include_rect.y() + j * dy

                # Use the include shape's containment method to check if the point is inside.
                if not include_shape.contains(include_shape.mapFromScene(x, y)):
                    continue

                # Check if the point falls in any exclude zone.
                skip = False
                for ex in exclude_shapes:
                    if ex.contains(ex.mapFromScene(x, y)):
                        skip = True
                        break
                if not skip:
                    points.append((x, y))

        # Draw the points on the image view.
        for (x, y) in points:
            # Draw a small blue circle (radius 2).
            pt_item = QGraphicsEllipseItem(x - 2, y - 2, 4, 4)
            pt_item.setBrush(QColor("blue"))
            pt_item.setPen(QPen(Qt.NoPen))
            self.image_view.scene.addItem(pt_item)
            self.image_view.generated_points.append(pt_item)

        # Update the points table.
        self.pointsTable.setRowCount(len(points))
        for idx, (x, y) in enumerate(points):
            self.pointsTable.setItem(idx, 0, QTableWidgetItem(str(idx + 1)))
            self.pointsTable.setItem(idx, 1, QTableWidgetItem(f"{x:.2f}"))
            self.pointsTable.setItem(idx, 2, QTableWidgetItem(f"{y:.2f}"))
            self.pointsTable.setItem(idx, 3, QTableWidgetItem(""))  # Placeholder for measurement.
