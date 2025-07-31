# zone_measurements/beam_center_utils.py

import json


def get_beam_center(poni, detector_size=(256, 256)):
    """
    Extracts beam center from PONI text and detector size.
    Returns (center_x, center_y) in pixel coordinates.
    """
    import re

    if not poni:
        # fallback to image center
        return (detector_size[0] // 2, detector_size[1] // 2)

    # Parse lines
    lines = poni.splitlines()
    poni1 = poni2 = pixel1 = pixel2 = None

    # First, extract Poni1 and Poni2 (in meters)
    for line in lines:
        if line.startswith("Poni1:"):
            try:
                poni1 = float(line.split(":")[1].strip())
            except Exception:
                pass
        if line.startswith("Poni2:"):
            try:
                poni2 = float(line.split(":")[1].strip())
            except Exception:
                pass
        if line.startswith("Detector_config:"):
            m = re.search(r"Detector_config:\s*(\{.*\})", line)
            if m:
                try:
                    cfg = json.loads(m.group(1))
                    pixel1 = float(cfg.get("pixel1"))
                    pixel2 = float(cfg.get("pixel2"))
                except Exception:
                    pass

    # Calculate center coordinates in pixels (col, row)
    if None not in (poni1, poni2, pixel1, pixel2):
        center_y = poni1 / pixel1  # row
        center_x = poni2 / pixel2  # col
        return (center_x, center_y)  # (x, y), both in pixels

    # Fallback to image center if any value missing
    return (detector_size[0] // 2, detector_size[1] // 2)
