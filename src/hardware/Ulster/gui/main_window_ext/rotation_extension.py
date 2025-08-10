from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QAction, QMenu, QToolButton


class RotatorToolButton(QToolButton):
    """
    A custom QToolButton that performs a rotation action.

    Left-click will rotate using a default angle.
    Right-click will display a menu to select a different rotation angle.
    """

    def __init__(self, text: str, default_angle: float, rotate_callback, parent=None):
        """
        Initialize the RotatorToolButton.

        Args:
            text (str): The text displayed on the button.
            default_angle (float): The initial rotation angle in degrees.
            rotate_callback (Callable): A callback function that performs the rotation.
            parent (QWidget, optional): The parent widget.
        """
        super().__init__(parent)
        self.setText(text)
        self.default_angle = default_angle  # Store the default rotation angle.
        self.rotate_callback = (
            rotate_callback  # Store the callback function to execute a rotation.
        )

        # Create a QMenu to allow the user to choose among different rotation angles.
        self.menu = QMenu(self)
        for angle in [0.5, 1, 2, 5, 10]:
            action = QAction(f"{angle}°", self)
            action.setData(angle)  # Store the numerical angle in the action.
            self.menu.addAction(action)
        # Connect the menu's triggered signal to update the default rotation angle.
        self.menu.triggered.connect(self.on_menu_triggered)

    def mousePressEvent(self, event):
        """
        Override mouse press event to check for right-click.

        A right-click displays the angle selection menu.
        For other mouse buttons, the default behavior is executed.

        Args:
            event (QMouseEvent): The mouse event.
        """
        if event.button() == Qt.RightButton:
            # Display the menu at the current global cursor position.
            self.menu.exec_(event.globalPos())
        else:
            # For non-right clicks, process the event normally.
            super().mousePressEvent(event)

    def on_menu_triggered(self, action: QAction):
        """
        Slot called when an action in the menu is triggered.

        Updates the default rotation angle to the one selected by the user.

        Args:
            action (QAction): The action that was triggered.
        """
        # Retrieve the angle from the action's stored data and update the default angle.
        angle = action.data()
        self.default_angle = angle

    def mouseReleaseEvent(self, event):
        """
        Override mouse release event to perform rotation.

        On a left-button release, the stored callback is invoked with the current default angle.

        Args:
            event (QMouseEvent): The mouse event.
        """
        if event.button() == Qt.LeftButton:
            # Trigger the rotation callback with the current default angle.
            self.rotate_callback(self.default_angle)
        # Ensure normal event processing for the remaining actions.
        super().mouseReleaseEvent(event)


class RotationMixin:
    """
    Mixin class to add image rotation functionality to a widget.

    This mixin assumes the widget provides:
    - A 'toolbar' attribute (e.g., QToolBar) to which action buttons will be added.
    - An 'image_view' attribute with a method 'rotate_image(angle)' to rotate the image.
    """

    def create_rotation_actions(self):
        """
        Create rotation action buttons for rotating the image left or right.
        """
        # Instantiate buttons with the default angle set to 1 degree.
        self.rotate_left_btn = RotatorToolButton(
            "Rotate Left", 1, self.rotate_left, self
        )
        self.rotate_right_btn = RotatorToolButton(
            "Rotate Right", 1, self.rotate_right, self
        )

    def add_rotation_actions_to_tool_bar(self):
        """
        Add the rotation buttons to the toolbar.
        """
        self.toolbar.addWidget(self.rotate_left_btn)
        self.toolbar.addWidget(self.rotate_right_btn)

    def rotate_left(self, angle: float):
        """
        Rotate the image to the left by the given angle.

        Args:
            angle (float): The rotation angle in degrees. A negative value rotates left.
        """
        # Pass a negative angle to rotate left.
        self.image_view.rotate_image(-angle)

    def rotate_right(self, angle: float):
        """
        Rotate the image to the right by the given angle.

        Args:
            angle (float): The rotation angle in degrees.
        """
        self.image_view.rotate_image(angle)
