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

def radial_profile_unit_conversion(radial_count=None,
        sample_distance_mm=None,
        wavelength_nm=None,
        pixel_size=None,
        radial_units="q_per_nm"):
    """
    Convert radial profile from pixel lengths to:
    - q_per_nm
    - two_theta
    - um

    Parameters
    ----------

    radial_count : int
        Number of radial points.

    sample_distance : float
        Meters.

    radial_units : str
        Choice of "q_per_nm" (default), "two_theta", or "um".
    """
    radial_range_m = np.arange(radial_count) * pixel_size
    radial_range_m = radial_range_m.reshape(-1,1)
    radial_range_mm = radial_range_m * 1e3

    if radial_units == "q_per_nm":
        q_range_per_nm = q_conversion(
            real_position_mm=radial_range_mm,
            sample_distance_mm=sample_distance_mm,
            wavelength_nm=wavelength_nm)
        return q_range_per_nm

    if radial_units == "two_theta":
        two_theta_range = two_theta_conversion(
                sample_distance_mm, radial_range_mm)
        return two_theta_range

    if radial_units == "um":
        return radial_range_m * 1e6

def two_theta_conversion(real_position_mm=None, sample_distance_mm=None):
    """
    Convert real position to two*theta
    """
    two_theta = np.arctan2(real_position_mm, sample_distance_mm)
    return two_theta

def q_conversion(
        real_position_mm=None, sample_distance_mm=None, wavelength_nm=None):
    """
    Convert real position to q
    """
    two_theta = two_theta_conversion(
            real_position_mm=real_position_mm,
            sample_distance_mm=sample_distance_mm)
    theta = two_theta / 2
    q_per_nm = 4*np.pi*np.sin(theta) / wavelength_nm
    return q_per_nm

def real_position_from_two_theta(two_theta=None, sample_distance_mm=None):
    """
    two_theta : float
        radians

    sample_distance_m
    """
    position_mm = sample_distance_mm * np.tan(two_theta)
    return position_mm

def real_position_from_q(q_per_nm=None, sample_distance_mm=None, wavelength_nm=None):
    """
    """
    theta = np.arcsin(q_per_nm * wavelength_nm / 4 / np.pi)
    two_theta = 2*theta
    position_mm = real_position_from_two_theta(
            two_theta=two_theta, sample_distance_mm=sample_distance_mm)
    return position_mm

def sample_distance_from_q(
        q_per_nm=None, wavelength_nm=None, real_position_mm=None):
    """
    """
    theta = np.arcsin(q_per_nm * wavelength_nm / (4*np.pi))
    two_theta = 2 * theta
    sample_distance_mm = real_position_mm / np.tan(two_theta)
    return sample_distance_mm
