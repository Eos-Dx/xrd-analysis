from typing import Dict, List, Optional, Union

import numpy as np
from scipy import fft
from scipy.integrate import simpson as simps

from .utility_functions import extract_image_data_values, mask_beam_center


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
    L = len(curve) - 1
    x = np.arange(0, L + 1)
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


def fourier_fft2(
    data: np.ndarray,
    remove_beam: str = "false",
    thresh: float = 700,
    padding: int = 0,
    mask: Optional[np.ndarray] = None,
    filter_radius: Optional[float] = None,
    features: Optional[Union[List[str], str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Perform FFT2 processing with selective feature extraction.

    :param data: Input 2D data array
    :param remove_beam: Beam removal method ('false', 'real', 'fourier')
    :param thresh: Threshold for beam removal
    :param padding: Padding around beam for removal
    :param filter_radius: Optional radius for frequency domain filtering
    :param features: Specific features to extract. Options include:
        - 'fft2_shifted': Shifted FFT
        - 'fft2_real': Real component
        - 'fft2_imag': Imaginary component
        - 'fft2_norm_magnitude': Normalized magnitude
        - 'fft2_phase': Phase
        - 'fft2_reconstructed': Reconstructed image
        - 'fft2_vertical_profile': Vertical frequency profile
        - 'fft2_horizontal_profile': Horizontal frequency profile
        - 'fft2_freq_horizontal': Frequency x-axis
        - 'fft2_freq_vertical': Frequency y-axis
        - 'all': Return all features (default)
    :return: Dictionary of selected FFT features
    """
    # Normalize input to lowercase
    remove_beam = remove_beam.lower()

    # Default to all features if not specified
    if features is None or features == "all":
        features = [
            "fft2_shifted",
            "fft2_real",
            "fft2_imag",
            "fft2_norm_magnitude",
            "fft2_magnitude",
            "fft2_phase",
            "fft2_reconstructed",
            "fft2_vertical_profile",
            "fft2_horizontal_profile",
            "fft2_freq_horizontal",
            "fft2_freq_vertical",
        ]
    elif isinstance(features, str):
        features = [features]

    # Beam removal in real space if specified
    if remove_beam == "real":
        if mask is not None:
            mask = extract_image_data_values(data, mask)
            data = data - mask
        else:
            beam = mask_beam_center(data, thresh, padding)
            data = data - beam

    # Compute FFT and shift
    fft2 = fft.fft2(data)
    fft2_shifted = fft.fftshift(fft2)

    # Beam removal in Fourier space if specified
    if remove_beam == "fourier":
        if mask is not None:
            mask = extract_image_data_values(data, mask)
            mask_fft = fft.fftshift(fft.fft2(mask))
            fft2_shifted = fft2_shifted - mask_fft
        else:
            beam = mask_beam_center(data, thresh, padding)
            beam_fft = fft.fftshift(fft.fft2(beam))
            fft2_shifted = fft2_shifted - beam_fft

    # Apply frequency filtering if radius is specified
    if filter_radius is not None:
        # Create circular mask
        rows, cols = data.shape
        crow, ccol = rows // 2, cols // 2
        Y, X = np.ogrid[:rows, :cols]
        mask_radius = (X - ccol) ** 2 + (Y - crow) ** 2 <= filter_radius**2
        fft2_shifted *= mask_radius

    # Prepare all possible features
    all_features = {
        "fft2_shifted": fft2_shifted,
        "fft2_real": np.real(fft2_shifted),
        "fft2_imag": np.imag(fft2_shifted),
        "fft2_norm_magnitude": np.divide(
            np.abs(fft2_shifted), np.abs(fft2[0, 0])
        ),
        "fft2_magnitude": np.abs(fft2_shifted),
        "fft2_phase": np.angle(fft2_shifted),
        "fft2_reconstructed": np.real(fft.ifft2(fft.ifftshift(fft2_shifted))),
        "fft2_vertical_profile": np.divide(
            np.abs(fft2_shifted), np.abs(fft2[0, 0])
        )[:, fft2_shifted.shape[1] // 2],
        "fft2_horizontal_profile": np.divide(
            np.abs(fft2_shifted), np.abs(fft2[0, 0])
        )[fft2_shifted.shape[0] // 2, :],
        "fft2_freq_horizontal": fft.fftshift(fft.fftfreq(data.shape[1])),
        "fft2_freq_vertical": fft.fftshift(fft.fftfreq(data.shape[0])),
    }

    # Return only requested features
    return {
        feature: all_features[feature]
        for feature in features
        if feature in all_features
    }
