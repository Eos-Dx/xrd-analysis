import random
import math
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QSpinBox, QPushButton, QTableWidget, QTableWidgetItem,
    QGraphicsEllipseItem, QLabel, QComboBox
)
from PyQt5.QtCore import Qt, QEvent, QPointF, QRectF
from PyQt5.QtGui import QColor, QPen, QTransform


class ZonePointsMixin:
    def createZonePointsWidget(self):
        """Creates a dock widget that generates evenly distributed points in the allowed region.
        The allowed region is defined as the 'include' zone shrunk by a user-defined percentage
        (default 5%) without changing its center, minus any 'exclude' zones.
        Generated points are drawn as a red circle (diameter 8) with a transparent cyan circle (20% opacity)
        underneath. User-defined points (added by double left-click) appear as larger blue circles.
        The table lists all points – first the generated ones, then the user-defined ones.
        Deletion via the table or by double right-click on a point removes it (for generated points, both the red
        item and its corresponding cyan circle are removed)."""
        self.zonePointsDock = QDockWidget("Zone Points", self)
        container = QWidget()
        layout = QVBoxLayout(container)

        # Input controls: number of points, percentage offset, and mm option control.
        inputLayout = QHBoxLayout()

        # Label and spin box for number of points.
        inputLayout.addWidget(QLabel("N points"))
        self.pointCountSpinBox = QSpinBox()
        self.pointCountSpinBox.setMinimum(2)
        self.pointCountSpinBox.setMaximum(1000)
        self.pointCountSpinBox.setValue(10)  # Default: 10 generated points.
        inputLayout.addWidget(self.pointCountSpinBox)

        # Label and spin box for percentage offset.
        inputLayout.addWidget(QLabel("% offset"))
        self.shrinkSpinBox = QSpinBox()
        self.shrinkSpinBox.setMinimum(0)
        self.shrinkSpinBox.setMaximum(100)
        self.shrinkSpinBox.setValue(5)  # Default: 5%
        inputLayout.addWidget(self.shrinkSpinBox)

        # Combo box for mm options.
        self.mmComboBox = QComboBox()
        for option in ["16mm", "14mm", "12mm", "10mm", "5mm", "4mm", "3mm", "2mm"]:
            self.mmComboBox.addItem(option)
        inputLayout.addWidget(self.mmComboBox)

        # Generate Points button.
        self.generatePointsBtn = QPushButton("Generate Points")
        inputLayout.addWidget(self.generatePointsBtn)
        layout.addLayout(inputLayout)

        # Table to display points.
        self.pointsTable = QTableWidget(0, 4)
        self.pointsTable.setHorizontalHeaderLabels(["ID", "X", "Y", "Measurement"])
        layout.addWidget(self.pointsTable)

        # Install event filter to catch Delete key presses on the table.
        self.pointsTable.installEventFilter(self)
        # Also install an event filter on the graphics view's viewport to catch double-click events.
        self.image_view.viewport().installEventFilter(self)

        container.setLayout(layout)
        self.zonePointsDock.setWidget(container)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.zonePointsDock)
        self.generatePointsBtn.clicked.connect(self.generateZonePoints)

        # Initialize lists:
        # generated_points (red) and generated_cyan (their corresponding cyan circles)
        # and user_defined_points (blue).
        self.image_view.generated_points = []
        self.image_view.generated_cyan = []
        self.user_defined_points = []

    def eventFilter(self, source, event):
        # Catch Delete key events on the table.
        if source == self.pointsTable and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Delete:
                self.deleteSelectedPoints()
                return True
        # Catch double-click events on the QGraphicsView's viewport.
        if source == self.image_view.viewport() and event.type() == QEvent.MouseButtonDblClick:
            self.mouseDoubleClickEvent(event)
            return True
        return super().eventFilter(source, event)

    def generateZonePoints(self):
        """
        Generate exactly N generated points (red, with cyan circles underneath) covering the allowed region
        using farthest–point sampling. The allowed region is defined as a shrunk version of the include zone.
        The shrink percentage is taken from self.shrinkSpinBox (default 5%),
        so that shrink_factor = (100 - shrink_percent) / 100.
        For a circular include zone (with attributes 'center' and 'radius'),
        the effective radius is: include_shape.radius * shrink_factor.
        For an arbitrary shape, its QPainterPath is scaled by shrink_factor about the center of its bounding rect.
        Candidate points (rejecting those in exclude zones) are sampled from the reduced domain and farthest–point
        sampling selects exactly N generated points.
        Before drawing new generated points, previously drawn generated points and cyan circles are removed.
        User-defined points (blue) remain.
        """
        N = self.pointCountSpinBox.value()
        shrink_percent = self.shrinkSpinBox.value()
        shrink_factor = (100 - shrink_percent) / 100.0

        # Retrieve include and exclude shapes.
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

        # Remove previously drawn generated items (both red and cyan) – leave user-defined ones.
        if hasattr(self.image_view, 'generated_points'):
            for item in self.image_view.generated_points:
                self.image_view.scene.removeItem(item)
            self.image_view.generated_points = []
        else:
            self.image_view.generated_points = []
        if hasattr(self.image_view, 'generated_cyan'):
            for item in self.image_view.generated_cyan:
                self.image_view.scene.removeItem(item)
            self.image_view.generated_cyan = []
        else:
            self.image_view.generated_cyan = []

        # Get bounding rectangle for candidate sampling.
        if hasattr(include_shape, "rect"):
            inc_rect = include_shape.rect()
        else:
            inc_rect = include_shape.boundingRect()
        x_min = inc_rect.x()
        y_min = inc_rect.y()
        x_max = x_min + inc_rect.width()
        y_max = y_min + inc_rect.height()

        candidates = []
        num_candidates = 10000
        attempts = 0

        def distance(p, q):
            return math.hypot(p[0] - q[0], p[1] - q[1])

        if hasattr(include_shape, 'center') and hasattr(include_shape, 'radius'):
            # If include zone is a circle, shrink its radius.
            inc_center = include_shape.center  # tuple (x,y)
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
            # For arbitrary include zone, shrink its QPainterPath.
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

        # Farthest–point sampling to choose exactly N generated points.
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

        # Draw each generated point:
        # 1. Draw a transparent cyan circle (20% opacity) with radius = ideal_radius.
        # 2. Draw a red circle (diameter = 8) on top.
        for (x, y) in final_points:
            cyan_item = QGraphicsEllipseItem(x - ideal_radius, y - ideal_radius,
                                             2 * ideal_radius, 2 * ideal_radius)
            cyan_color = QColor("cyan")
            cyan_color.setAlphaF(0.2)
            cyan_item.setBrush(cyan_color)
            cyan_item.setPen(QPen(Qt.NoPen))
            self.image_view.scene.addItem(cyan_item)
            self.image_view.generated_cyan.append(cyan_item)

            red_item = QGraphicsEllipseItem(x - 4, y - 4, 8, 8)
            red_item.setBrush(QColor("red"))
            red_item.setPen(QPen(Qt.NoPen))
            red_item.setFlags(QGraphicsEllipseItem.ItemIsSelectable | QGraphicsEllipseItem.ItemIsMovable)
            red_item.setData(0, "generated")
            self.image_view.scene.addItem(red_item)
            self.image_view.generated_points.append(red_item)

        self.updatePointsTable()

    def updatePointsTable(self):
        """
        Update the table to list all points.
        The table shows generated points (red) first and then user-defined points (blue).
        """
        points = []
        # Append generated points.
        if hasattr(self.image_view, 'generated_points'):
            for item in self.image_view.generated_points:
                center = item.sceneBoundingRect().center()
                points.append((center.x(), center.y(), "generated"))
        # Append user-defined points.
        if hasattr(self, "user_defined_points"):
            for item in self.user_defined_points:
                center = item.sceneBoundingRect().center()
                points.append((center.x(), center.y(), "user"))
        self.pointsTable.setRowCount(len(points))
        for idx, (x, y, ptype) in enumerate(points):
            self.pointsTable.setItem(idx, 0, QTableWidgetItem(str(idx + 1)))
            self.pointsTable.setItem(idx, 1, QTableWidgetItem(f"{x:.2f}"))
            self.pointsTable.setItem(idx, 2, QTableWidgetItem(f"{y:.2f}"))
            self.pointsTable.setItem(idx, 3, QTableWidgetItem(ptype))

    def deleteSelectedPoints(self):
        """
        Deletes points selected in the table.
        For generated points (red), both the red item and its corresponding cyan circle (at the same table index)
        are removed. For user-defined points (blue), the blue item is removed.
        The table is then updated.
        """
        selected_indexes = self.pointsTable.selectionModel().selectedRows()
        # We assume the table rows are ordered: first generated, then user-defined.
        generated_count = len(self.image_view.generated_points) if hasattr(self.image_view, 'generated_points') else 0
        # Process rows in descending order.
        rows = sorted([index.row() for index in selected_indexes], reverse=True)
        for row in rows:
            if row < generated_count:
                # Remove from generated lists.
                red_item = self.image_view.generated_points.pop(row)
                self.image_view.scene.removeItem(red_item)
                if hasattr(self.image_view, 'generated_cyan') and row < len(self.image_view.generated_cyan):
                    cyan_item = self.image_view.generated_cyan.pop(row)
                    self.image_view.scene.removeItem(cyan_item)
            else:
                # Row belongs to user-defined points.
                user_index = row - generated_count
                if hasattr(self, "user_defined_points") and user_index < len(self.user_defined_points):
                    blue_item = self.user_defined_points.pop(user_index)
                    self.image_view.scene.removeItem(blue_item)
        self.updatePointsTable()

    def mouseDoubleClickEvent(self, event):
        # Use the graphics view's mapToScene method.
        if event.button() == Qt.LeftButton:
            pos = self.image_view.mapToScene(event.pos())
            blue_radius = 10  # local radius for blue points
            # Create ellipse with local coordinates centered at (0,0)
            blue_item = QGraphicsEllipseItem(-blue_radius, -blue_radius, 2 * blue_radius, 2 * blue_radius)
            blue_item.setBrush(QColor("blue"))
            blue_item.setPen(QPen(Qt.NoPen))
            blue_item.setFlags(QGraphicsEllipseItem.ItemIsSelectable | QGraphicsEllipseItem.ItemIsMovable)
            blue_item.setData(0, "user")
            blue_item.setPos(pos)
            self.image_view.scene.addItem(blue_item)
            if not hasattr(self, "user_defined_points"):
                self.user_defined_points = []
            self.user_defined_points.append(blue_item)
            self.updatePointsTable()
            event.accept()
        elif event.button() == Qt.RightButton:
            pos = self.image_view.mapToScene(event.pos())
            tolerance = 5  # pixels
            rect = QRectF(pos.x() - tolerance, pos.y() - tolerance, 2 * tolerance, 2 * tolerance)
            items = self.image_view.scene.items(rect)
            indices_to_delete = set()
            for item in items:
                # Check for generated items (red or its cyan circle).
                if item.data(0) in ["generated", "generated_cyan"]:
                    if hasattr(self.image_view, "generated_points") and item in self.image_view.generated_points:
                        idx = self.image_view.generated_points.index(item)
                        indices_to_delete.add(idx)
                    elif hasattr(self.image_view, "generated_cyan") and item in self.image_view.generated_cyan:
                        idx = self.image_view.generated_cyan.index(item)
                        indices_to_delete.add(idx)
                # Also check for user-defined points.
                elif item.data(0) == "user":
                    if hasattr(self, "user_defined_points") and item in self.user_defined_points:
                        self.user_defined_points.remove(item)
                        self.image_view.scene.removeItem(item)
            if indices_to_delete:
                for idx in sorted(indices_to_delete, reverse=True):
                    red_item = self.image_view.generated_points.pop(idx)
                    self.image_view.scene.removeItem(red_item)
                    if hasattr(self.image_view, "generated_cyan") and idx < len(self.image_view.generated_cyan):
                        cyan_item = self.image_view.generated_cyan.pop(idx)
                        self.image_view.scene.removeItem(cyan_item)
            self.updatePointsTable()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)
