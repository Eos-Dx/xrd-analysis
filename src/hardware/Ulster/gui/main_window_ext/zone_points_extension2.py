import random
import math
import numpy as np
from shapely.geometry import Point, Polygon
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QSpinBox, QPushButton, QTableWidget, QTableWidgetItem, QGraphicsEllipseItem
)
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QColor, QPen

class ZonePointsMixin:
    def createZonePointsWidget(self):
        """Creates a dock widget that allows generating points evenly distributed
        in the allowed region (include minus exclude) using a Monte Carlo simulated annealing approach.
        Points can be deleted by selecting rows and pressing the Delete key."""
        self.zonePointsDock = QDockWidget("Zone Points", self)
        container = QWidget()
        layout = QVBoxLayout(container)

        # Input control for the number of points and generate button.
        inputLayout = QHBoxLayout()
        self.pointCountSpinBox = QSpinBox()
        self.pointCountSpinBox.setMinimum(2)
        self.pointCountSpinBox.setMaximum(1000)
        self.pointCountSpinBox.setValue(10)  # Default: 10 points.
        inputLayout.addWidget(self.pointCountSpinBox)

        self.generatePointsBtn = QPushButton("Generate Points")
        inputLayout.addWidget(self.generatePointsBtn)
        layout.addLayout(inputLayout)

        # Table to show generated points.
        self.pointsTable = QTableWidget(0, 4)
        self.pointsTable.setHorizontalHeaderLabels(["ID", "X", "Y", "Measurement"])
        layout.addWidget(self.pointsTable)

        # Install an event filter to catch Delete key presses.
        self.pointsTable.installEventFilter(self)

        container.setLayout(layout)
        self.zonePointsDock.setWidget(container)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.zonePointsDock)

        self.generatePointsBtn.clicked.connect(self.generateZonePoints)

    def eventFilter(self, source, event):
        # When the Delete key is pressed while the pointsTable is focused, delete the selected points.
        if source == self.pointsTable and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Delete:
                self.deleteSelectedPoints()
                return True
        return super().eventFilter(source, event)

    def generateZonePoints(self):
        """
        Generate exactly N points covering the allowed region (include minus exclude)
        using a Monte Carlo simulated annealing approach. The allowed region is defined
        by an include shape and one or more exclude shapes. A margin is enforced so that no point
        is placed closer than a specified distance from any boundary.
        """
        N = self.pointCountSpinBox.value()

        # Retrieve the include and exclude shapes from self.image_view.shapes.
        # Each shape is expected to have a "role" key ("include" or "exclude") and an "item"
        # key containing a Shapely geometry (or an object with attributes like 'center' and 'radius').
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

        # --- Define a helper to compute the allowed domain.
        def allowed_domain(include, excludes, margin):
            # Buffer the include shape inward by the margin and subtract each exclude shape.
            domain = include.buffer(-margin)
            for ex in excludes:
                domain = domain.difference(ex)
            return domain

        # Set margin: if include_shape has an attribute 'radius' (circle), use a fraction of it.
        margin = 0.01 * include_shape.radius if hasattr(include_shape, 'radius') else 0.01
        domain = allowed_domain(include_shape, exclude_shapes, margin)
        if domain.is_empty:
            print("Allowed domain is empty after applying margin/excludes.")
            return

        # --- Define helper functions.
        def random_point_in_domain(domain):
            # Use rejection sampling based on the domain's bounds.
            minx, miny, maxx, maxy = domain.bounds
            while True:
                x = random.uniform(minx, maxx)
                y = random.uniform(miny, maxy)
                p = Point(x, y)
                if domain.contains(p):
                    return np.array([x, y])

        def distance(p, q):
            return math.hypot(p[0] - q[0], p[1] - q[1])

        def cost_function(points):
            # Cost is defined as the variance of the minimum distance between each point and its nearest neighbor.
            dists = []
            for i, p in enumerate(points):
                distances = [distance(p, q) for j, q in enumerate(points) if i != j]
                dists.append(min(distances))
            return np.var(dists)

        def simulated_annealing(N, iterations=10000, T_init=1.0, T_final=0.001):
            points = [random_point_in_domain(domain) for _ in range(N)]
            current_cost = cost_function(points)
            T = T_init
            for i in range(iterations):
                # Exponential cooling
                T = T_init * ((T_final / T_init) ** (i / iterations))
                idx = random.randint(0, N - 1)
                new_point = random_point_in_domain(domain)
                new_points = points.copy()
                new_points[idx] = new_point
                new_cost = cost_function(new_points)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / T):
                    points = new_points
                    current_cost = new_cost
            return points

        optimized_points = simulated_annealing(N)
        print("Optimized points:", optimized_points)

        # --- Optionally, adjust points to ensure they maintain the margin from boundaries.
        def adjust_point(pt):
            new_pt = pt
            # Adjust relative to include shape boundary if it has center and radius attributes.
            if hasattr(include_shape, 'center') and hasattr(include_shape, 'radius'):
                d_inc = distance(pt, include_shape.center)
                if include_shape.radius - d_inc < margin:
                    # Move inward so that the new distance is include_shape.radius - margin.
                    if d_inc > 0:
                        scale = (include_shape.radius - margin) / d_inc
                        new_pt = (include_shape.center[0] + (pt[0] - include_shape.center[0]) * scale,
                                  include_shape.center[1] + (pt[1] - include_shape.center[1]) * scale)
            # Adjust relative to each exclude shape.
            for ex in exclude_shapes:
                if hasattr(ex, 'center') and hasattr(ex, 'radius'):
                    d_ex = distance(new_pt, ex.center)
                    if d_ex - ex.radius < margin:
                        # Move outward so that new distance is ex.radius + margin.
                        if d_ex > 0:
                            scale = (ex.radius + margin) / d_ex
                            new_pt = (ex.center[0] + (new_pt[0] - ex.center[0]) * scale,
                                      ex.center[1] + (new_pt[1] - ex.center[1]) * scale)
            return new_pt

        adjusted_points = [adjust_point(pt) for pt in optimized_points]

        # --- Draw the points on the scene and update the table.
        if hasattr(self.image_view, 'generated_points'):
            for pt_item in self.image_view.generated_points:
                self.image_view.scene.removeItem(pt_item)
            self.image_view.generated_points = []
        else:
            self.image_view.generated_points = []

        for (x, y) in adjusted_points:
            pt_item = QGraphicsEllipseItem(x - 2, y - 2, 4, 4)
            pt_item.setBrush(QColor("blue"))
            pt_item.setPen(QPen(Qt.NoPen))
            pt_item.setFlags(QGraphicsEllipseItem.ItemIsSelectable | QGraphicsEllipseItem.ItemIsMovable)
            self.image_view.scene.addItem(pt_item)
            self.image_view.generated_points.append(pt_item)
        self.updatePointsTable(adjusted_points)

    def updatePointsTable(self, points):
        self.pointsTable.setRowCount(len(points))
        for idx, (x, y) in enumerate(points):
            self.pointsTable.setItem(idx, 0, QTableWidgetItem(str(idx + 1)))
            self.pointsTable.setItem(idx, 1, QTableWidgetItem(f"{x:.2f}"))
            self.pointsTable.setItem(idx, 2, QTableWidgetItem(f"{y:.2f}"))
            self.pointsTable.setItem(idx, 3, QTableWidgetItem(""))

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
        new_points = []
        for pt in self.image_view.generated_points:
            rect = pt.rect()
            x = rect.x() + rect.width() / 2
            y = rect.y() + rect.height() / 2
            new_points.append((x, y))
        self.updatePointsTable(new_points)
