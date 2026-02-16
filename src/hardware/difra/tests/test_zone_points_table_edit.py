"""Tests for editing manual point coordinates directly in the points table."""

import os
import sys
import types
import unittest


# Ensure src root is importable
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


# Minimal PyQt5 stubs for importing zone_points_extension in headless unit tests
if "PyQt5" not in sys.modules:
    pyqt5 = types.ModuleType("PyQt5")
    qtcore = types.ModuleType("PyQt5.QtCore")
    qtgui = types.ModuleType("PyQt5.QtGui")
    qtwidgets = types.ModuleType("PyQt5.QtWidgets")
    sip_mod = types.ModuleType("PyQt5.sip")

    class _Qt:
        ItemIsSelectable = 1
        ItemIsEnabled = 2
        ItemIsEditable = 4
        Key_Delete = 16777223

    class _QEvent:
        KeyPress = 6

    class _QColor:
        def __init__(self, *args, **kwargs):
            pass

    class _WidgetPlaceholder:
        def __init__(self, *args, **kwargs):
            pass

    class _QTableWidgetItem:
        def __init__(self, text=""):
            self._text = str(text)
            self._flags = _Qt.ItemIsSelectable | _Qt.ItemIsEnabled | _Qt.ItemIsEditable
            self._row = -1
            self._col = -1

        def text(self):
            return self._text

        def setText(self, value):
            self._text = str(value)

        def setFlags(self, flags):
            self._flags = flags

        def flags(self):
            return self._flags

        def row(self):
            return self._row

        def column(self):
            return self._col

    qtcore.Qt = _Qt()
    qtcore.QEvent = _QEvent
    qtgui.QColor = _QColor
    qtwidgets.QDockWidget = _WidgetPlaceholder
    qtwidgets.QHBoxLayout = _WidgetPlaceholder
    qtwidgets.QLabel = _WidgetPlaceholder
    qtwidgets.QSplitter = _WidgetPlaceholder
    qtwidgets.QTableWidgetItem = _QTableWidgetItem
    qtwidgets.QTreeWidget = _WidgetPlaceholder
    qtwidgets.QTreeWidgetItem = _WidgetPlaceholder
    qtwidgets.QVBoxLayout = _WidgetPlaceholder
    qtwidgets.QWidget = _WidgetPlaceholder

    sip_mod.isdeleted = lambda obj: bool(getattr(obj, "_deleted", False))

    pyqt5.QtCore = qtcore
    pyqt5.QtGui = qtgui
    pyqt5.QtWidgets = qtwidgets
    pyqt5.sip = sip_mod

    sys.modules["PyQt5"] = pyqt5
    sys.modules["PyQt5.QtCore"] = qtcore
    sys.modules["PyQt5.QtGui"] = qtgui
    sys.modules["PyQt5.QtWidgets"] = qtwidgets
    sys.modules["PyQt5.sip"] = sip_mod


# Keep import lightweight by stubbing measurement widget dependency
if "hardware.difra.gui.technical.widgets" not in sys.modules:
    widgets_mod = types.ModuleType("hardware.difra.gui.technical.widgets")

    class _MeasurementHistoryWidget:
        def __init__(self, *args, **kwargs):
            pass

    widgets_mod.MeasurementHistoryWidget = _MeasurementHistoryWidget
    sys.modules["hardware.difra.gui.technical.widgets"] = widgets_mod


from hardware.difra.gui.main_window_ext.zone_points_extension import ZonePointsMixin


class _FakePoint:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def x(self):
        return self._x

    def y(self):
        return self._y


class _FakeRect:
    def __init__(self, x, y, w, h):
        self._x = x
        self._y = y
        self._w = w
        self._h = h

    def width(self):
        return self._w

    def height(self):
        return self._h

    def center(self):
        return _FakePoint(self._x + self._w / 2.0, self._y + self._h / 2.0)


class _FakeEllipseItem:
    def __init__(self, center_x, center_y, radius, point_id=None):
        self._rect = _FakeRect(
            center_x - radius,
            center_y - radius,
            2.0 * radius,
            2.0 * radius,
        )
        self._point_id = point_id
        self._deleted = False

    def rect(self):
        return self._rect

    def setRect(self, x, y, w, h):
        self._rect = _FakeRect(x, y, w, h)

    def data(self, role):
        if role == 1:
            return self._point_id
        return None

    def sceneBoundingRect(self):
        return self._rect


class _FakeScene:
    def __init__(self):
        self.update_calls = 0

    def update(self):
        self.update_calls += 1


class _FakeTableItem:
    def __init__(self, row, col, text):
        self._row = row
        self._col = col
        self._text = str(text)

    def row(self):
        return self._row

    def column(self):
        return self._col

    def text(self):
        return self._text

    def setText(self, value):
        self._text = str(value)


class _FakeTable:
    def __init__(self):
        self._items = {}

    def setItem(self, row, col, item):
        self._items[(row, col)] = item

    def item(self, row, col):
        return self._items.get((row, col))


class _Harness(ZonePointsMixin):
    def __init__(self):
        self._updating_points_table = False
        self.pointsTable = _FakeTable()
        self.image_view = types.SimpleNamespace(
            points_dict={
                "generated": {"points": [], "zones": []},
                "user": {"points": [], "zones": []},
            },
            scene=_FakeScene(),
        )
        self.update_calls = 0

    def update_points_table(self):
        self.update_calls += 1


class TestZonePointsTableEdit(unittest.TestCase):
    def test_edit_user_point_moves_point_and_zone(self):
        harness = _Harness()
        user_point = _FakeEllipseItem(center_x=10.0, center_y=20.0, radius=4.0, point_id=7)
        user_zone = _FakeEllipseItem(center_x=10.0, center_y=20.0, radius=10.0)
        harness.image_view.points_dict["user"]["points"].append(user_point)
        harness.image_view.points_dict["user"]["zones"].append(user_zone)

        harness.pointsTable.setItem(0, 0, _FakeTableItem(0, 0, "7"))
        harness.pointsTable.setItem(0, 1, _FakeTableItem(0, 1, "42.50"))
        harness.pointsTable.setItem(0, 2, _FakeTableItem(0, 2, "20.00"))

        changed_item = harness.pointsTable.item(0, 1)
        harness.on_points_table_item_changed(changed_item)

        self.assertAlmostEqual(user_point.sceneBoundingRect().center().x(), 42.5)
        self.assertAlmostEqual(user_point.sceneBoundingRect().center().y(), 20.0)
        self.assertAlmostEqual(user_zone.sceneBoundingRect().center().x(), 42.5)
        self.assertAlmostEqual(user_zone.sceneBoundingRect().center().y(), 20.0)
        self.assertEqual(harness.image_view.scene.update_calls, 1)
        self.assertEqual(harness.update_calls, 1)

    def test_edit_generated_point_is_rejected_and_restored(self):
        harness = _Harness()
        generated_point = _FakeEllipseItem(
            center_x=15.0, center_y=25.0, radius=4.0, point_id=9
        )
        harness.image_view.points_dict["generated"]["points"].append(generated_point)

        harness.pointsTable.setItem(0, 0, _FakeTableItem(0, 0, "9"))
        harness.pointsTable.setItem(0, 1, _FakeTableItem(0, 1, "50.00"))
        harness.pointsTable.setItem(0, 2, _FakeTableItem(0, 2, "25.00"))

        harness.on_points_table_item_changed(harness.pointsTable.item(0, 1))

        self.assertAlmostEqual(generated_point.sceneBoundingRect().center().x(), 15.0)
        self.assertAlmostEqual(generated_point.sceneBoundingRect().center().y(), 25.0)
        self.assertEqual(harness.update_calls, 1)
        self.assertEqual(harness.image_view.scene.update_calls, 0)

    def test_invalid_coordinate_keeps_user_point_unchanged(self):
        harness = _Harness()
        user_point = _FakeEllipseItem(center_x=8.0, center_y=9.0, radius=4.0, point_id=11)
        harness.image_view.points_dict["user"]["points"].append(user_point)
        harness.image_view.points_dict["user"]["zones"].append(None)

        harness.pointsTable.setItem(0, 0, _FakeTableItem(0, 0, "11"))
        harness.pointsTable.setItem(0, 1, _FakeTableItem(0, 1, "oops"))
        harness.pointsTable.setItem(0, 2, _FakeTableItem(0, 2, "9.00"))

        harness.on_points_table_item_changed(harness.pointsTable.item(0, 1))

        self.assertAlmostEqual(user_point.sceneBoundingRect().center().x(), 8.0)
        self.assertAlmostEqual(user_point.sceneBoundingRect().center().y(), 9.0)
        self.assertEqual(harness.update_calls, 1)


if __name__ == "__main__":
    unittest.main()
