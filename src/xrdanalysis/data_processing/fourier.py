import numpy as np
from scipy import fft
from scipy.integrate import simpson as simps

from .utility_functions import mask_beam_center


def slope_removal_custom(curve):
    """
    Removes the slope from a curve by rotating the curve such that the first \
    and last values are set to zero.

    :param curve: The input curve that requires slope removal.
    :type curve: numpy.ndarray
    :return: A tuple containing the flattened curve and the transformed x \
    values.
    :rtype: tuple (numpy.ndarray, numpy.ndarray)
    """
    curve = curve - curve[0]
    curve = list(curve)
    length = len(curve)
    x_range = np.zeros(length)
    curve_flat = np.zeros(length)
    angle = np.arctan(curve[-1] / (length - 1))
    for j in range(length):
        curve_flat[j] = curve[j] * np.cos(angle) - j * np.sin(angle)
        x_range[j] = j * np.cos(angle) + curve[j] * np.sin(angle)
    return curve_flat, x_range


def slope_removal(y):
    """
    Removes the slope of a curve by subtracting a linear function that \
    connects the first and last values.

    :param y: The input curve where slope needs to be removed.
    :type y: numpy.ndarray
    :return: The curve with slope removed.
    :rtype: numpy.ndarray
    """
    # Number of points
    n = len(y)

    # Get first and last values
    y1 = y[0]
    yn = y[-1]

    # Create a linear function passing through (1, y1) and (n, yn)
    linear_func = np.linspace(y1, yn, n)

    # Subtract the linear function from the original curve
    adjusted_y = y - linear_func

    return adjusted_y


def fourier_custom(curve, order):
    """
    Applies a custom Fourier transform to extract the Fourier coefficients up \
    to a given order.

    :param curve: The input curve to transform using Fourier series.
    :type curve: numpy.ndarray
    :param order: The number of Fourier terms (harmonics) to consider in the \
    expansion.
    :type order: int
    :return: Concatenated list of 'an' (cosine) and 'bn' (sine) coefficients.
    :rtype: numpy.ndarray
    """
    L = len(curve)
    x = np.arange(0, L)
    a0 = 2.0 / L * simps(curve, x)
    an_list = np.zeros(order)
    bn_list = np.zeros(order)

    for n in range(1, order + 1):
        cos_term = np.cos(2 * np.pi * n * x / L)
        sin_term = np.sin(2 * np.pi * n * x / L)

        an_list[n - 1] = 2.0 / L * simps(curve * cos_term, x)
        bn_list[n - 1] = 2.0 / L * simps(curve * sin_term, x)

    inverse = a0 / 2.0 + sum(
        [
            an_list[k - 1] * np.cos(2.0 * np.pi * k * x / L)
            + bn_list[k - 1] * np.sin(2.0 * np.pi * k * x / L)
            for k in range(1, order + 1)
        ]
    )

    return np.concatenate([an_list, bn_list]), inverse


def fourier_fft(curve, order):
    """
    Applies Fourier transform using FFT (Fast Fourier Transform) and extracts \
    the real and imaginary coefficients.

    :param curve: The input curve to transform using FFT.
    :type curve: numpy.ndarray
    :param order: The number of Fourier terms (harmonics) to consider in the \
    expansion.
    :type order: int
    :return: Concatenated list of real and imaginary parts of the Fourier \
    coefficients.
    :rtype: numpy.ndarray
    """
    # 0 order is ignored
    fourier = fft.fft(curve)

    coeff = fourier[1 : order + 1]  # noqa: E203

    filtered_fourier = np.array(fourier.copy())
    filtered_fourier[order + 1 : -order] = 0  # noqa: E203

    inverse = fft.ifft(filtered_fourier)
    return np.concatenate([coeff.real, coeff.imag]), inverse


def compute_fft2_magnitude(
    data: np.ndarray, remove_beam=False, thresh=700, padding=0
) -> np.ndarray:
    """
    Compute FFT2 magnitude with optional beam removal.

    :param data: Input 2D array
    :return: Normalized magnitude of FFT2
    """
    # Compute initial FFT
    fft2 = fft.fft2(data)
    fft2_shifted = fft.fftshift(fft2)

    if remove_beam:
        # Remove beam in real space
        beam = mask_beam_center(data, thresh, padding)
        beam_fft = fft.fft2(beam)
        beam_fft_shifted = fft.fftshift(beam_fft)
        # Subtract beam in Fourier space
        fft2_shifted = fft2_shifted - beam_fft_shifted

    # Calculate and normalize magnitude
    magnitude = np.abs(fft2_shifted)
    magnitude_norm = np.divide(magnitude, np.abs(fft2[0, 0]))

    return magnitude_norm, fft2, fft2_shifted


def apply_fft2_filter(data, radius, remove_beam=False, thresh=700, padding=0):
    # Step 1: Perform FFT and shift the zero-frequency component to the center
    fft2 = fft.fft2(data)
    fft2_shifted = fft.fftshift(fft2)

    if remove_beam:
        # Remove beam in real space
        beam = mask_beam_center(data, thresh, padding)
        beam_fft = fft.fft2(beam)
        beam_fft_shifted = fft.fftshift(beam_fft)
        # Subtract beam in Fourier space
        fft2_shifted = fft2_shifted - beam_fft_shifted

    # Step 2: Create a circular mask with the given radius
    rows, cols = data.shape
    crow, ccol = rows // 2, cols // 2  # center of the image
    Y, X = np.ogrid[:rows, :cols]
    mask = (X - ccol) ** 2 + (Y - crow) ** 2 <= radius**2

    # Step 3: Apply the mask to the shifted FFT image
    filtered_fft = fft2_shifted * mask

    # Step 4: Extract real, imaginary, magnitude, and phase components
    real_component = np.real(filtered_fft)
    imaginary_component = np.imag(filtered_fft)
    magnitude_component = np.abs(filtered_fft)
    magnitude_norm = np.divide(magnitude_component, np.abs(fft2[0, 0]))
    phase_component = np.angle(filtered_fft)

    # Compute frequency axes
    freq_x = fft.fftshift(fft.fftfreq(data.shape[1]))
    freq_y = fft.fftshift(fft.fftfreq(data.shape[0]))

    # Get frequency profiles
    vertical_profile = magnitude_norm[:, magnitude_norm.shape[1] // 2]
    horizontal_profile = magnitude_norm[magnitude_norm.shape[0] // 2, :]

    # Shift back before inverse transform
    fft2_unshifted = fft.ifftshift(filtered_fft)
    reconstructed = np.real(fft.ifft2(fft2_unshifted))

    return (
        fft2_shifted,
        real_component,
        imaginary_component,
        magnitude_norm,
        phase_component,
        reconstructed,
        vertical_profile,
        horizontal_profile,
        freq_x,
        freq_y,
    )


def apply_fft2(
    data,
    remove_beam=False,
    thresh=700,
    padding=0,
    batch_normalize: bool = False,
    batch_mean: float = None,
    batch_std: float = None,
):
    magnitude_norm, fft2, fft2_shifted = compute_fft2_magnitude(
        data, remove_beam, thresh, padding
    )
    fft2_real = np.real(fft2_shifted)
    fft2_imag = np.imag(fft2_shifted)

    # Apply batch normalization if enabled
    if batch_normalize:
        magnitude_norm = (magnitude_norm - batch_mean) / batch_std

    phase = np.angle(fft2_shifted)

    # Compute frequency axes
    freq_x = fft.fftshift(fft.fftfreq(data.shape[1]))
    freq_y = fft.fftshift(fft.fftfreq(data.shape[0]))

    # Get frequency profiles
    vertical_profile = magnitude_norm[:, magnitude_norm.shape[1] // 2]
    horizontal_profile = magnitude_norm[magnitude_norm.shape[0] // 2, :]

    # Compute inverse transform
    if remove_beam:
        # Shift back before inverse transform
        fft2_unshifted = fft.ifftshift(fft2_shifted)
        reconstructed = np.real(fft.ifft2(fft2_unshifted))
    else:
        reconstructed = np.real(fft.ifft2(fft2))

    return (
        fft2_shifted,
        fft2_real,
        fft2_imag,
        magnitude_norm,
        phase,
        reconstructed,
        vertical_profile,
        horizontal_profile,
        freq_x,
        freq_y,
    )


def fourier_fft2(
    data: np.ndarray,
    remove_beam: bool = False,
    thresh: float = 1000,
    padding: int = 0,
    batch_normalize: bool = False,
    batch_mean: float = None,
    batch_std: float = None,
    filter_radius: int = None,
) -> dict:
    """
    Performs 2D Fourier analysis with optional beam removal.

    :param data: Input 2D data array.
    :type data: np.ndarray
    :param remove_beam: Whether to remove central beam.
    :type remove_beam: bool, optional
    :param thresh: Threshold for beam removal.
    :type thresh: float, optional
    :param padding: Padding around beam for removal.
    :type padding: int, optional
    :returns: Dictionary containing Fourier analysis results.
    :rtype: dict
    """

    if filter_radius and not batch_normalize:
        (
            fft2_shifted,
            fft2_real,
            fft2_imag,
            magnitude_norm,
            phase,
            reconstructed,
            vertical_profile,
            horizontal_profile,
            freq_x,
            freq_y,
        ) = apply_fft2_filter(
            data, filter_radius, remove_beam, thresh, padding
        )

    else:
        (
            fft2_shifted,
            fft2_real,
            fft2_imag,
            magnitude_norm,
            phase,
            reconstructed,
            vertical_profile,
            horizontal_profile,
            freq_x,
            freq_y,
        ) = apply_fft2(
            data,
            remove_beam,
            thresh,
            padding,
            batch_normalize,
            batch_mean,
            batch_std,
        )

    return {
        "fft2_shifted": fft2_shifted,
        "fft2_real": fft2_real,
        "fft2_imag": fft2_imag,
        "fft2_norm_magnitude": magnitude_norm,
        "fft2_phase": phase,
        "fft2_reconstructed": reconstructed,
        "fft2_vertical_profile": vertical_profile,
        "fft2_horizontal_profile": horizontal_profile,
        "fft2_freq_horizontal": freq_x,
        "fft2_freq_vertical": freq_y,
    }
