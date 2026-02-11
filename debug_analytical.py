#!/usr/bin/env python3
"""Debug script to check analytical measurement creation"""

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

# Add project src to path
SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hardware.container.v0_1 import schema, writer
from hardware.container.v0_1.technical_container import (
    create_technical_container,
    write_detector_config,
    write_pony_datasets,
)
from hardware.container.v0_1.container_manager import lock_container

# Create temp directory
temp_dir = Path(tempfile.mkdtemp())
print(f"Working in: {temp_dir}")

# Create minimal technical container
tech_id, tech_path = create_technical_container(
    folder=temp_dir,
    distance_cm=17.0,
)

# Add detector config
detector_config = [
    {
        "id": "DET1",
        "alias": "DET1",
        "type": "AdvaPIX",
        "size": [256, 256],
        "pixel_size_um": 55.0,
    },
]

write_detector_config(tech_path, detector_config, ["DET1"])

# Add PONI data
poni_content = """Detector: AdvaPIX
PixelSize1: 5.500e-05
PixelSize2: 5.500e-05
Distance: 0.170000
Poni1: 0.014025
Poni2: 0.014025
Rot1: 0.000000
Rot2: 0.000000
Rot3: 0.000000
Detector_config: {"pixel1": 5.5e-05, "pixel2": 5.5e-05, "max_shape": [256, 256]}
"""

pony_data = {
    "DET1": (poni_content, "DET1_17cm.poni"),
}

write_pony_datasets(tech_path, pony_data, 17.0)

# Lock technical container
lock_container(tech_path)

# Create session container
session_id, session_path = writer.create_session_container(
    folder=temp_dir,
    sample_id="SAMPLE_001",
    operator_id="test_operator",
    site_id="test_site",
    machine_name="DIFRA-01",
    beam_energy_keV=17.5,
    acquisition_date="2024-01-15",
)

# Copy technical data to session
writer.copy_technical_to_session(
    technical_file=tech_path,
    session_file=session_path,
    auto_lock=False,  # Already locked
)

print(f"Session container: {session_path}")

# Check session structure
with h5py.File(session_path, "r") as f:
    print("\nSession structure:")
    f.visititems(lambda name, obj: print(f"  /{name}"))

# Add attenuation measurement
attenuation_data = {
    "DET1": np.random.randint(100, 1000, size=(256, 256), dtype=np.uint16),
}

detector_metadata = {
    "DET1": {
        "integration_time_ms": 50.0,
        "beam_energy_keV": 17.5,
    },
}

pony_alias_map = {"DET1": "DET1"}

print("\nAdding analytical measurement...")
ana_path = writer.add_analytical_measurement(
    file_path=session_path,
    measurement_data=attenuation_data,
    detector_metadata=detector_metadata,
    pony_alias_map=pony_alias_map,
    analysis_type="attenuation",
    timestamp_start="2024-01-15 10:30:00",
)

print(f"Analytical measurement path: {ana_path}")

# Check structure again
with h5py.File(session_path, "r") as f:
    print("\nSession structure after analytical measurement:")
    f.visititems(lambda name, obj: print(f"  /{name}"))
    
    print(f"\nChecking {ana_path}:")
    if ana_path in f:
        ana_group = f[ana_path]
        print(f"  Attributes: {dict(ana_group.attrs)}")
        print(f"  Children: {list(ana_group.keys())}")
    else:
        print(f"  NOT FOUND in file!")

print(f"\nTest files in: {temp_dir}")
