import random
import math
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QSpinBox, QPushButton, QTableWidget, QTableWidgetItem, QGraphicsEllipseItem
)
from PyQt5.QtCore import Qt, QEvent, QPointF
from PyQt5.QtGui import QColor, QPen, QTransform


class ZonePointsMixin:
    def createZonePointsWidget(self):
        """Creates a dock widget that generates evenly distributed points in the allowed region.
        The allowed region is defined as the 'include' zone shrunk by a user-controlled percentage
        (default 5%) without changing its center, minus any 'exclude' zones.
        Each point is drawn with a transparent cyan circle (20% opacity) covering the ideal area,
        and a larger red circle (diameter 8) on top.
        Points can be deleted by selecting rows and pressing the Delete key."""
        self.zonePointsDock = QDockWidget("Zone Points", self)
        container = QWidget()
        layout = QVBoxLayout(container)

        # Input controls: one for the number of points and one for the shrink percentage.
        inputLayout = QHBoxLayout()
        self.pointCountSpinBox = QSpinBox()
        self.pointCountSpinBox.setMinimum(2)
        self.pointCountSpinBox.setMaximum(1000)
        self.pointCountSpinBox.setValue(10)  # Default: 10 points.
        inputLayout.addWidget(self.pointCountSpinBox)

        # New control: shrink percentage for the include zone.
        self.shrinkSpinBox = QSpinBox()
        self.shrinkSpinBox.setMinimum(0)
        self.shrinkSpinBox.setMaximum(100)
        self.shrinkSpinBox.setValue(5)  # Default: 5%
        inputLayout.addWidget(self.shrinkSpinBox)

        self.generatePointsBtn = QPushButton("Generate Points")
        inputLayout.addWidget(self.generatePointsBtn)
        layout.addLayout(inputLayout)

        # Table to display generated points.
        self.pointsTable = QTableWidget(0, 4)
        self.pointsTable.setHorizontalHeaderLabels(["ID", "X", "Y", "Measurement"])
        layout.addWidget(self.pointsTable)

        # Install event filter to catch Delete key presses.
        self.pointsTable.installEventFilter(self)

        container.setLayout(layout)
        self.zonePointsDock.setWidget(container)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.zonePointsDock)
        self.generatePointsBtn.clicked.connect(self.generateZonePoints)

    def eventFilter(self, source, event):
        if source == self.pointsTable and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Delete:
                self.deleteSelectedPoints()
                return True
        return super().eventFilter(source, event)

    def generateZonePoints(self):
        """
        Generate exactly N points covering the allowed region (include minus exclude)
        using farthest–point sampling.

        The allowed region is defined as a shrunk version of the include zone. The shrink percentage
        is defined by the user (default 5%), so that the effective domain is scaled by:
            shrink_factor = (100 - shrink_percent)/100.0.

        For a circular include zone (with attributes 'center' and 'radius'),
        the reduced radius is: include_shape.radius * shrink_factor.
        For an arbitrary shape, its QPainterPath is scaled by shrink_factor about the center of its bounding rectangle.

        Candidate points are sampled from the reduced domain (rejecting points in exclude zones),
        and farthest–point sampling selects exactly N points.

        Finally, for each final point a transparent cyan circle (with fill opacity 20%) of radius equal to
        the ideal radius (computed from allowed area per point) is drawn, with a red circle (diameter 8)
        drawn on top.
        """
        N = self.pointCountSpinBox.value()
        shrink_percent = self.shrinkSpinBox.value()
        shrink_factor = (100 - shrink_percent) / 100.0

        # Retrieve the include shape and any exclude shapes.
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

        # Get a bounding rectangle for candidate sampling.
        if hasattr(include_shape, "rect"):
            inc_rect = include_shape.rect()
        else:
            inc_rect = include_shape.boundingRect()
        x_min = inc_rect.x()
        y_min = inc_rect.y()
        x_max = x_min + inc_rect.width()
        y_max = y_min + inc_rect.height()

        # Prepare candidate sampling from the "shrunk" domain.
        candidates = []
        num_candidates = 10000
        attempts = 0

        # Helper: Euclidean distance.
        def distance(p, q):
            return math.hypot(p[0] - q[0], p[1] - q[1])

        if hasattr(include_shape, 'center') and hasattr(include_shape, 'radius'):
            # For a circle: reduce its radius by the shrink factor.
            inc_center = include_shape.center  # tuple (x, y)
            reduced_radius = include_shape.radius * shrink_factor

            def sample_from_circle(center, radius):
                angle = random.uniform(0, 2 * math.pi)
                r = math.sqrt(random.random()) * radius
                return (center[0] + r * math.cos(angle), center[1] + r * math.sin(angle))

            while len(candidates) < num_candidates and attempts < num_candidates * 10:
                attempts += 1
                pt = sample_from_circle(inc_center, reduced_radius)
                if not include_shape.contains(include_shape.mapFromScene(pt[0], pt[1])):
                    continue
                in_exclude = False
                for ex in exclude_shapes:
                    if ex.contains(ex.mapFromScene(pt[0], pt[1])):
                        in_exclude = True
                        break
                if in_exclude:
                    continue
                candidates.append(pt)
        else:
            # For an arbitrary shape: obtain its QPainterPath and compute a shrunk version.
            path = include_shape.shape()
            bounding = include_shape.boundingRect()
            center_point = bounding.center()
            transform = QTransform()
            transform.translate(center_point.x(), center_point.y())
            transform.scale(shrink_factor, shrink_factor)
            transform.translate(-center_point.x(), -center_point.y())
            shrunk_path = transform.map(path)
            while len(candidates) < num_candidates and attempts < num_candidates * 10:
                attempts += 1
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
                if not shrunk_path.contains(QPointF(x, y)):
                    continue
                in_exclude = False
                for ex in exclude_shapes:
                    if ex.contains(ex.mapFromScene(x, y)):
                        in_exclude = True
                        break
                if in_exclude:
                    continue
                candidates.append((x, y))
        if not candidates:
            print("No candidate points found in the shrunk allowed region.")
            return

        # Estimate allowed area from candidate density.
        bbox_area = (x_max - x_min) * (y_max - y_min)
        allowed_area = bbox_area * (len(candidates) / float(attempts))
        circle_area = allowed_area / N
        ideal_radius = math.sqrt(circle_area / math.pi)
        print(f"Allowed area: {allowed_area:.2f}, ideal radius: {ideal_radius:.2f}")

        # Farthest–point sampling: start with the candidate closest to the centroid.
        sum_x = sum(pt[0] for pt in candidates)
        sum_y = sum(pt[1] for pt in candidates)
        centroid = (sum_x / len(candidates), sum_y / len(candidates))
        initial = min(candidates, key=lambda pt: distance(pt, centroid))
        chosen = [initial]
        candidates.remove(initial)
        while len(chosen) < N and candidates:
            best_candidate = None
            best_dist = -1
            for cand in candidates:
                d = min(distance(cand, p) for p in chosen)
                if d > best_dist:
                    best_dist = d
                    best_candidate = cand
            if best_candidate is None:
                break
            chosen.append(best_candidate)
            candidates.remove(best_candidate)
        final_points = chosen[:N]

        # Draw the final points.
        # For each point, draw:
        #  - A transparent cyan circle (20% fill) with radius equal to ideal_radius.
        #  - A red circle (diameter 8) on top.
        if hasattr(self.image_view, 'generated_points'):
            for item in self.image_view.generated_points:
                self.image_view.scene.removeItem(item)
            self.image_view.generated_points = []
        else:
            self.image_view.generated_points = []

        for (x, y) in final_points:
            # Draw transparent cyan circle.
            cyan_item = QGraphicsEllipseItem(x - ideal_radius, y - ideal_radius, 2 * ideal_radius, 2 * ideal_radius)
            cyan_color = QColor("cyan")
            cyan_color.setAlphaF(0.2)  # 20% opacity
            cyan_item.setBrush(cyan_color)
            cyan_item.setPen(QPen(Qt.NoPen))
            self.image_view.scene.addItem(cyan_item)
            # Draw red circle on top (diameter 8).
            red_item = QGraphicsEllipseItem(x - 4, y - 4, 8, 8)
            red_item.setBrush(QColor("red"))
            red_item.setPen(QPen(Qt.NoPen))
            red_item.setFlags(QGraphicsEllipseItem.ItemIsSelectable | QGraphicsEllipseItem.ItemIsMovable)
            self.image_view.scene.addItem(red_item)
            # Store only the red item as the representative point.
            self.image_view.generated_points.append(red_item)

        self.updatePointsTable(final_points)

    def updatePointsTable(self, points):
        self.pointsTable.setRowCount(len(points))
        for idx, (x, y) in enumerate(points):
            self.pointsTable.setItem(idx, 0, QTableWidgetItem(str(idx + 1)))
            self.pointsTable.setItem(idx, 1, QTableWidgetItem(f"{x:.2f}"))
            self.pointsTable.setItem(idx, 2, QTableWidgetItem(f"{y:.2f}"))
            self.pointsTable.setItem(idx, 3, QTableWidgetItem(""))

    def deleteSelectedPoints(self):
        selected_indexes = self.pointsTable.selectionModel().selectedRows()
        rows = [index.row() for index in selected_indexes]
        rows.sort(reverse=True)
        for row in rows:
            try:
                if hasattr(self.image_view, 'generated_points') and row < len(self.image_view.generated_points):
                    item = self.image_view.generated_points[row]
                    self.image_view.scene.removeItem(item)
                    del self.image_view.generated_points[row]
            except Exception as e:
                print("Error deleting point:", e)
        new_points = []
        for item in self.image_view.generated_points:
            rect = item.rect()
            x = rect.x() + rect.width() / 2
            y = rect.y() + rect.height() / 2
            new_points.append((x, y))
        self.updatePointsTable(new_points)
