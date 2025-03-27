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

        # --- Restore zone points using the unified dictionary ---
        zone_points = state.get("zone_points", [])
        # Clear any existing points in the unified dictionary.
        if hasattr(self.image_view, "points_dict"):
            for key in ["generated", "user"]:
                for pt in self.image_view.points_dict[key]["points"]:
                    self.image_view.scene.removeItem(pt)
                for zone in self.image_view.points_dict[key]["zones"]:
                    self.image_view.scene.removeItem(zone)
                self.image_view.points_dict[key]["points"] = []
                self.image_view.points_dict[key]["zones"] = []
        else:
            self.image_view.points_dict = {
                "generated": {"points": [], "zones": []},
                "user": {"points": [], "zones": []}
            }

        for pt in zone_points:
            x = pt.get("x")
            y = pt.get("y")
            pt_type = pt.get("type")
            if pt_type == "user":
                from PyQt5.QtWidgets import QGraphicsEllipseItem
                blue_radius = 10
                # Create the blue marker.
                user_marker = QGraphicsEllipseItem(-blue_radius, -blue_radius, 2 * blue_radius, 2 * blue_radius)
                user_marker.setBrush(QColor("blue"))
                user_marker.setPen(QPen())
                user_marker.setFlags(QGraphicsEllipseItem.ItemIsSelectable | QGraphicsEllipseItem.ItemIsMovable)
                user_marker.setData(0, "user")
                user_marker.setPos(x, y)
                self.image_view.scene.addItem(user_marker)
                self.image_view.points_dict["user"]["points"].append(user_marker)
                # Create the associated zone.
                zone_item = QGraphicsEllipseItem(x - blue_radius, y - blue_radius, 2 * blue_radius, 2 * blue_radius)
                zone_color = QColor("blue")
                zone_color.setAlphaF(0.2)
                zone_item.setBrush(zone_color)
                zone_item.setPen(QPen())
                self.image_view.scene.addItem(zone_item)
                self.image_view.points_dict["user"]["zones"].append(zone_item)
            elif pt_type == "generated":
                from PyQt5.QtWidgets import QGraphicsEllipseItem
                # Create the red marker.
                red_marker = QGraphicsEllipseItem(x - 4, y - 4, 8, 8)
                red_marker.setBrush(QColor("red"))
                red_marker.setPen(QPen())
                red_marker.setFlags(QGraphicsEllipseItem.ItemIsSelectable | QGraphicsEllipseItem.ItemIsMovable)
                red_marker.setData(0, "generated")
                self.image_view.scene.addItem(red_marker)
                self.image_view.points_dict["generated"]["points"].append(red_marker)
                # Create the cyan zone.
                cyan_zone = QGraphicsEllipseItem(x - 5, y - 5, 10, 10)
                cyan_color = QColor("cyan")
                cyan_color.setAlphaF(0.2)
                cyan_zone.setBrush(cyan_color)
                cyan_zone.setPen(QPen())
                self.image_view.scene.addItem(cyan_zone)
                self.image_view.points_dict["generated"]["zones"].append(cyan_zone)

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
        if hasattr(self.image_view, "points_dict"):
            # Save generated points.
            for pt in self.image_view.points_dict["generated"]["points"]:
                center = pt.sceneBoundingRect().center()
                zone_points.append({"x": center.x(), "y": center.y(), "type": "generated"})
            # Save user-defined points.
            for pt in self.image_view.points_dict["user"]["points"]:
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
        if hasattr(self.image_view, "points_dict"):
            for pt in self.image_view.points_dict["generated"]["points"]:
                center = pt.sceneBoundingRect().center()
                zone_points.append({"x": center.x(), "y": center.y(), "type": "generated"})
            for pt in self.image_view.points_dict["user"]["points"]:
                center = pt.sceneBoundingRect().center()
                zone_points.append({"x": center.x(), "y": center.y(), "type": "user"})
        state["zone_points"] = zone_points

        try:
            with open(self.PREV_STATE_FILE, "w") as f:
                json.dump(state, f, indent=4)
            print("State manually saved to", self.PREV_STATE_FILE)
        except Exception as e:
            print("Error manually saving state:", e)

    def setupAutoSave(self, interval=2000):
        """
        Sets up a QTimer to automatically save the state every 'interval' milliseconds.
        """
        self.autoSaveTimer = QTimer(self)
        self.autoSaveTimer.timeout.connect(self.autoSaveState)
        self.autoSaveTimer.start(interval)
