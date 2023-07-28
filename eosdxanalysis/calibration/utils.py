def relative_beam_center(
        beam_center : tuple = None,
        pixel_size : float = None,
        pixel_count_x : int = None,
        detector_sensor_spacing : float = None,
        ):
    """Calculate relative beam center from detector 2
    given the beam center location in detector 1
    in row, column format.

    Note
    ----
    ``pixel_size`` and ``detector_sensor_spacing`` must have the same length
    units.
    """
    relative_beam_center_col = \
            -(pixel_count_x - beam_center[1]) - \
            detector_sensor_spacing/pixel_size
    relative_beam_center_row = beam_center[0]
    result = relative_beam_center_row, relative_beam_center_col
    return result
