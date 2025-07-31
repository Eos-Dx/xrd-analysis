# zone_measurements/measurement_table_utils.py


def add_measurement_to_table(
    points_table, row, measurement_widget, results, timestamp
):
    """
    Attaches a MeasurementHistoryWidget to the specified row in the table and updates it.
    """
    if not isinstance(measurement_widget, object):
        # Instantiate your widget if not present
        # measurement_widget = MeasurementHistoryWidget(...)
        pass
    points_table.setCellWidget(row, 5, measurement_widget)
    measurement_widget.add_measurement(results, timestamp)


def delete_selected_points(points_table, measurement_widgets, selected_rows):
    """
    Removes selected rows from the table and deletes associated widgets.
    Optionally, deletes measurement files to avoid data confusion.
    """
    for row in sorted(selected_rows, reverse=True):
        widget = measurement_widgets.get(row)
        if widget is not None:
            # Optionally, delete widget data/files here
            widget.deleteLater()
            measurement_widgets.pop(row, None)
        points_table.removeRow(row)
