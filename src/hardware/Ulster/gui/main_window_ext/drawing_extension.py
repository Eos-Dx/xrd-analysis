from PyQt5.QtWidgets import QAction, QActionGroup


class DrawingMixin:
    """
    Mixin class that adds drawing mode actions to a UI component.

    This mixin expects the using class to have:
      - a 'toolbar' attribute (e.g., a QToolBar) to which actions are added.
      - an 'image_view' attribute that provides:
          * set_drawing_mode(mode: Optional[str]) to switch modes.
          * delete_selected_shapes() to remove currently selected shapes.
    """

    def create_drawing_actions(self) -> None:
        """
        Create QAction objects for different drawing modes and group them together.

        Modes included:
          - Rectangle ("rect"): for drawing rectangular shapes.
          - Circle ("ellipse"): for drawing elliptical shapes.
          - Crop ("crop"): for cropping the image.
          - Select (None): to disable drawing and enable selection.

        The 'Select' mode is checked by default.
        """
        # Create actions with checkable states and assign each its callback.
        self.select_rect_act = QAction("Rectangle", self, checkable=True, triggered=self.select_rect_mode)
        self.select_ellipse_act = QAction("Circle", self, checkable=True, triggered=self.select_ellipse_mode)
        self.crop_act = QAction("Crop", self, checkable=True, triggered=self.select_crop_mode)
        self.select_act = QAction("Select", self, checkable=True, triggered=self.select_select_mode)

        # Group actions to ensure only one mode is active at a time.
        self.drawingModeGroup = QActionGroup(self)
        self.drawingModeGroup.addAction(self.select_rect_act)
        self.drawingModeGroup.addAction(self.select_ellipse_act)
        self.drawingModeGroup.addAction(self.crop_act)
        self.drawingModeGroup.addAction(self.select_act)

        # Set the "Select" mode as the default.
        self.select_act.setChecked(True)

    def add_drawing_actions_to_tool_bar(self) -> None:
        """
        Add drawing actions to the toolbar.
        """
        # Add each drawing mode action to the toolbar.
        self.toolbar.addAction(self.select_rect_act)
        self.toolbar.addAction(self.select_ellipse_act)
        self.toolbar.addAction(self.crop_act)
        self.toolbar.addAction(self.select_act)

    # Mode switching callback methods:
    def select_rect_mode(self) -> None:
        """
        Switch to rectangle drawing mode.
        """
        # Tell the image view to set its drawing mode to rectangle.
        self.image_view.set_drawing_mode("rect")

    def select_ellipse_mode(self) -> None:
        """
        Switch to ellipse (circle) drawing mode.
        """
        # Tell the image view to set its drawing mode to ellipse.
        self.image_view.set_drawing_mode("ellipse")

    def select_crop_mode(self) -> None:
        """
        Switch to crop mode.
        """
        # Tell the image view to set its drawing mode to crop.
        self.image_view.set_drawing_mode("crop")

    def select_select_mode(self) -> None:
        """
        Switch to selection mode, disabling any drawing functionality.
        """
        # Set the drawing mode to None, which typically represents the selection mode.
        self.image_view.set_drawing_mode(None)

    def create_delete_action(self) -> None:
        """
        Create an action to delete selected shapes.
        """
        # Create a delete action that calls the 'delete_selected_shapes' method when triggered.
        self.delete_act = QAction("Delete", self, triggered=self.delete_selected_shapes)

    def add_delete_action_to_tool_bar(self) -> None:
        """
        Add the delete action to the toolbar.
        """
        # Add the delete action to the toolbar.
        self.toolbar.addAction(self.delete_act)

    def delete_selected_shapes(self) -> None:
        """
        Delete the shapes currently selected in the image view.

        Checks if the image_view attribute exists before attempting deletion.
        """
        if hasattr(self, 'image_view'):
            self.image_view.delete_selected_shapes()
