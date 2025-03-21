import os
import json
import shutil
from PyQt5.QtCore import QTimer, QRectF
from PyQt5.QtGui import QPen, QColor

class StateSaverMixin:
    AUTO_STATE_FILE = "autosave_state.json"
    PREV_STATE_FILE = "autosave_state_prev.json"

    def restoreState(self):
        """
        Restores the state from the previous state file (PREV_STATE_FILE).
        If that file is not available, it tries to load the current autosave file.
        """
        state_file = None
        if os.path.exists(self.PREV_STATE_FILE):
            state_file = self.PREV_STATE_FILE
        elif os.path.exists(self.AUTO_STATE_FILE):
            state_file = self.AUTO_STATE_FILE

        if not state_file:
            print("No saved state file found. Nothing to restore.")
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)
        except Exception as e:
            print("Error loading saved state from", state_file, ":", e)
            return

        # --- Restore image ---
        image_path = state.get("image")
        if image_path:
            from PyQt5.QtGui import QPixmap
            pixmap = QPixmap(image_path)
            self.image_view.setImage(pixmap, image_path=image_path)
        else:
            print("No image to restore.")

        # --- Restore rotation ---
        if "rotation_angle" in state:
            angle = state["rotation_angle"]
            self.image_view.rotation_angle = angle
            if self.image_view.image_item:
                self.image_view.image_item.setRotation(angle)

        # --- Restore crop rectangle ---
        if state.get("crop_rect"):
            rect = state["crop_rect"]
            self.image_view.crop_rect = QRectF(rect['x'], rect['y'], rect['width'], rect['height'])
        else:
            self.image_view.crop_rect = None

        # --- Restore shapes ---
        shapes = state.get("shapes", [])
        if shapes:
            if hasattr(self.image_view, "shapes"):
                for shape_info in self.image_view.shapes:
                    item = shape_info.get("item")
                    if item:
                        self.image_view.scene.removeItem(item)
                self.image_view.shapes = []
            else:
                self.image_view.shapes = []
            for shape in shapes:
                shape_id = shape.get("id")
                shape_type = shape.get("type")
                role = shape.get("role", "include")
                geometry = shape.get("geometry")
                x, y = geometry.get("x"), geometry.get("y")
                w, h = geometry.get("width"), geometry.get("height")
                if shape_type.lower() in ["rect", "rectangle"]:
                    from PyQt5.QtWidgets import QGraphicsRectItem
                    item = QGraphicsRectItem(x, y, w, h)
                elif shape_type.lower() in ["ellipse", "circle"]:
                    from PyQt5.QtWidgets import QGraphicsEllipseItem
                    item = QGraphicsEllipseItem(x, y, w, h)
                else:
                    from PyQt5.QtWidgets import QGraphicsRectItem
                    item = QGraphicsRectItem(x, y, w, h)
                pen = QPen(QColor("green") if role == "include" else QColor("red"), 2)
                item.setPen(pen)
                self.image_view.scene.addItem(item)
                self.image_view.shapes.append({
                    "id": shape_id,
                    "type": shape_type,
                    "role": role,
                    "item": item
                })

        # --- Restore zone points ---
        zone_points = state.get("zone_points", [])
        if hasattr(self.image_view, "generated_points"):
            for pt in self.image_view.generated_points:
                self.image_view.scene.removeItem(pt)
            self.image_view.generated_points = []
        if hasattr(self.image_view, "generated_cyan"):
            for pt in self.image_view.generated_cyan:
                self.image_view.scene.removeItem(pt)
            self.image_view.generated_cyan = []
        if hasattr(self, "user_defined_points"):
            for pt in self.user_defined_points:
                self.image_view.scene.removeItem(pt)
            self.user_defined_points = []
        else:
            self.user_defined_points = []

        for pt in zone_points:
            x = pt.get("x")
            y = pt.get("y")
            pt_type = pt.get("type")
            if pt_type == "user":
                from PyQt5.QtWidgets import QGraphicsEllipseItem
                blue_radius = 10
                item = QGraphicsEllipseItem(-blue_radius, -blue_radius, 2 * blue_radius, 2 * blue_radius)
                item.setBrush(QColor("blue"))
                item.setPen(QPen())
                item.setFlags(QGraphicsEllipseItem.ItemIsSelectable | QGraphicsEllipseItem.ItemIsMovable)
                item.setData(0, "user")
                item.setPos(x, y)
                self.image_view.scene.addItem(item)
                self.user_defined_points.append(item)
            elif pt_type == "generated":
                from PyQt5.QtWidgets import QGraphicsEllipseItem
                red_item = QGraphicsEllipseItem(x - 4, y - 4, 8, 8)
                red_item.setBrush(QColor("red"))
                red_item.setPen(QPen())
                red_item.setFlags(QGraphicsEllipseItem.ItemIsSelectable | QGraphicsEllipseItem.ItemIsMovable)
                red_item.setData(0, "generated")
                self.image_view.scene.addItem(red_item)
                self.image_view.generated_points.append(red_item)
                cyan_item = QGraphicsEllipseItem(x - 5, y - 5, 10, 10)
                cyan_color = QColor("cyan")
                cyan_color.setAlphaF(0.2)
                cyan_item.setBrush(cyan_color)
                cyan_item.setPen(QPen())
                self.image_view.scene.addItem(cyan_item)
                self.image_view.generated_cyan.append(cyan_item)

        if hasattr(self, "updateShapeTable"):
            self.updateShapeTable()
        if hasattr(self, "updatePointsTable"):
            self.updatePointsTable()

    def autoSaveState(self):
        """
        Collects the current state and saves it to AUTO_STATE_FILE.
        Before saving, copies the current autosave file to PREV_STATE_FILE (if it exists).
        """
        state = {}
        state["image"] = getattr(self.image_view, "current_image_path", None)
        state["rotation_angle"] = getattr(self.image_view, "rotation_angle", 0)
        if hasattr(self.image_view, "crop_rect") and self.image_view.crop_rect is not None:
            rect = self.image_view.crop_rect
            state["crop_rect"] = {
                "x": rect.x(),
                "y": rect.y(),
                "width": rect.width(),
                "height": rect.height()
            }
        else:
            state["crop_rect"] = None

        shapes = []
        if hasattr(self.image_view, "shapes"):
            for shape in self.image_view.shapes:
                item = shape.get("item")
                if item:
                    rect = item.sceneBoundingRect()
                    shapes.append({
                        "id": shape.get("id"),
                        "type": shape.get("type"),
                        "role": shape.get("role", "include"),
                        "geometry": {
                            "x": rect.x(),
                            "y": rect.y(),
                            "width": rect.width(),
                            "height": rect.height()
                        }
                    })
        state["shapes"] = shapes

        zone_points = []
        if hasattr(self.image_view, "generated_points"):
            for pt in self.image_view.generated_points:
                center = pt.sceneBoundingRect().center()
                zone_points.append({"x": center.x(), "y": center.y(), "type": "generated"})
        if hasattr(self, "user_defined_points"):
            for pt in self.user_defined_points:
                center = pt.sceneBoundingRect().center()
                zone_points.append({"x": center.x(), "y": center.y(), "type": "user"})
        state["zone_points"] = zone_points

        # Before saving new autosave, copy existing autosave file (if any) to PREV_STATE_FILE.
        if os.path.exists(self.AUTO_STATE_FILE):
            try:
                shutil.copyfile(self.AUTO_STATE_FILE, self.PREV_STATE_FILE)
            except Exception as e:
                print("Error copying autosave file to previous state file:", e)

        try:
            with open(self.AUTO_STATE_FILE, "w") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            print("Error saving state:", e)

    def manualSaveState(self):
        """
        Manually collects the current state and saves it directly to PREV_STATE_FILE.
        This method is intended to be triggered by a toolbar button.
        """
        state = {}
        state["image"] = getattr(self.image_view, "current_image_path", None)
        state["rotation_angle"] = getattr(self.image_view, "rotation_angle", 0)
        if hasattr(self.image_view, "crop_rect") and self.image_view.crop_rect is not None:
            rect = self.image_view.crop_rect
            state["crop_rect"] = {
                "x": rect.x(),
                "y": rect.y(),
                "width": rect.width(),
                "height": rect.height()
            }
        else:
            state["crop_rect"] = None

        shapes = []
        if hasattr(self.image_view, "shapes"):
            for shape in self.image_view.shapes:
                item = shape.get("item")
                if item:
                    rect = item.sceneBoundingRect()
                    shapes.append({
                        "id": shape.get("id"),
                        "type": shape.get("type"),
                        "role": shape.get("role", "include"),
                        "geometry": {
                            "x": rect.x(),
                            "y": rect.y(),
                            "width": rect.width(),
                            "height": rect.height()
                        }
                    })
        state["shapes"] = shapes

        zone_points = []
        if hasattr(self.image_view, "generated_points"):
            for pt in self.image_view.generated_points:
                center = pt.sceneBoundingRect().center()
                zone_points.append({"x": center.x(), "y": center.y(), "type": "generated"})
        if hasattr(self, "user_defined_points"):
            for pt in self.user_defined_points:
                center = pt.sceneBoundingRect().center()
                zone_points.append({"x": center.x(), "y": center.y(), "type": "user"})
        state["zone_points"] = zone_points

        try:
            with open(self.PREV_STATE_FILE, "w") as f:
                json.dump(state, f, indent=4)
            print("State manually saved to", self.PREV_STATE_FILE)
        except Exception as e:
            print("Error manually saving state:", e)

    def setupAutoSave(self, interval=60000):
        """
        Sets up a QTimer to automatically save the state every 'interval' milliseconds.
        """
        self.autoSaveTimer = QTimer(self)
        self.autoSaveTimer.timeout.connect(self.autoSaveState)
        self.autoSaveTimer.start(interval)
