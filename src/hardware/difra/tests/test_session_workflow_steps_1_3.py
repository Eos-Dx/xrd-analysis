"""Session workflow tests for steps 1-3 (create session, sample image, zones)."""

import os
import sys

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from _session_workflow_complete_shared import *  # noqa: F401,F403
def test_step1_create_session(temp_dir, technical_container_path):
    """Step 1: Create new session container with metadata."""
    print("\n=== STEP 1: Create Session Container ===")

    # Create session folder
    session_folder = temp_dir / "sessions"
    session_folder.mkdir()

    # Create session
    session_id, session_path = session_container.create_session_container(
        folder=session_folder,
        sample_id="TEST_SAMPLE_001",
        operator_id="test_operator",
        site_id="LAB_A",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-11",
        patient_id="PATIENT_XYZ",
    )

    print(f"  ✓ Session created: {Path(session_path).name}")
    print(f"  ✓ Session ID: {session_id}")
    print(f"  ✓ Sample ID: TEST_SAMPLE_001")

    # Verify session container structure
    assert Path(session_path).exists(), "Session file should exist"

    with h5py.File(session_path, "r") as f:
        # Check root attributes
        assert f.attrs["container_type"] == schema.CONTAINER_TYPE_SESSION
        assert f.attrs["session_id"] == session_id
        assert f.attrs["sample_id"] == "TEST_SAMPLE_001"
        assert f.attrs["operator_id"] == "test_operator"
        assert f.attrs["patient_id"] == "PATIENT_XYZ"
        assert f.attrs["beam_energy_keV"] == 17.5

        # Check required groups exist
        assert "/images" in f
        assert "/images/zones" in f
        assert "/images/mapping" in f
        assert "/points" in f
        assert "/measurements" in f
        assert "/analytical_measurements" in f

        print(f"  ✓ All required groups created")
        print(f"  ✓ Root attributes verified")

    # Copy technical data
    session_container.copy_technical_to_session(
        technical_file=technical_container_path, session_file=session_path
    )

    # Verify technical data was copied
    with h5py.File(session_path, "r") as f:
        assert "/technical" in f
        assert "/technical/poni" in f
        print(f"  ✓ Technical calibration data copied")

        # Count technical events
        tech_group = f["/technical"]
        tech_events = [k for k in tech_group.keys() if k.startswith("tech_evt_")]
        print(f"  ✓ Technical events: {len(tech_events)}")

# ============================================================================
# Test 2: Add Sample Image
# ============================================================================


def test_step2_add_sample_image(temp_dir, technical_container_path):
    """Step 2: Add sample image to session container."""
    print("\n=== STEP 2: Add Sample Image ===")

    # Create session first
    session_folder = temp_dir / "sessions"
    session_folder.mkdir()

    session_id, session_path = session_container.create_session_container(
        folder=session_folder,
        sample_id="TEST_SAMPLE_002",
        operator_id="test_operator",
        site_id="LAB_A",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-11",
    )

    session_container.copy_technical_to_session(
        technical_file=technical_container_path, session_file=session_path
    )

    # Create synthetic sample image (grayscale)
    image_height, image_width = 800, 600
    image_data = np.random.randint(0, 255, size=(image_height, image_width), dtype=np.uint8)

    # Add image to session
    image_path = session_container.add_image(
        file_path=session_path,
        image_index=1,
        image_data=image_data,
        image_type="sample",
    )

    print(f"  ✓ Sample image added: {image_path}")
    print(f"  ✓ Image size: {image_width}x{image_height}")

    # Verify image stored correctly
    with h5py.File(session_path, "r") as f:
        assert "/images/img_001" in f
        img_group = f["/images/img_001"]
        assert "data" in img_group

        stored_data = img_group["data"][:]
        assert stored_data.shape == (image_height, image_width)
        assert np.array_equal(stored_data, image_data)

        assert img_group.attrs["image_type"] == "sample"
        print(f"  ✓ Image data integrity verified")

# ============================================================================
# Test 3: Define Measurement Zones
# ============================================================================


def test_step3_define_zones(temp_dir, technical_container_path):
    """Step 3: Define measurement zones (sample holder, include, exclude)."""
    print("\n=== STEP 3: Define Measurement Zones ===")

    # Setup session with image
    session_folder = temp_dir / "sessions"
    session_folder.mkdir()

    session_id, session_path = session_container.create_session_container(
        folder=session_folder,
        sample_id="TEST_SAMPLE_003",
        operator_id="test_operator",
        site_id="LAB_A",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-11",
    )

    session_container.copy_technical_to_session(
        technical_file=technical_container_path, session_file=session_path
    )

    image_data = np.random.randint(0, 255, size=(800, 600), dtype=np.uint8)
    session_container.add_image(
        file_path=session_path, image_index=1, image_data=image_data
    )

    # Define sample holder zone (circular)
    holder_geometry = [300.0, 400.0, 200.0, 200.0]  # x, y, width, height
    holder_diameter_mm = 25.4  # 1 inch

    zone1_path = session_container.add_zone(
        file_path=session_path,
        zone_index=1,
        zone_role="sample_holder",
        geometry_px=holder_geometry,
        shape="circle",
        holder_diameter_mm=holder_diameter_mm,
    )

    print(f"  ✓ Sample holder zone: {zone1_path}")
    print(f"    - Shape: circle")
    print(f"    - Diameter: {holder_diameter_mm} mm")

    # Define include zone (polygon)
    include_geometry = [
        [250.0, 350.0],
        [550.0, 350.0],
        [550.0, 650.0],
        [250.0, 650.0],
    ]

    zone2_path = session_container.add_zone(
        file_path=session_path,
        zone_index=2,
        zone_role="include",
        geometry_px=include_geometry,
        shape="polygon",
    )

    print(f"  ✓ Include zone: {zone2_path}")
    print(f"    - Shape: polygon")
    print(f"    - Vertices: {len(include_geometry)}")

    # Define exclude zone (circular - e.g., air bubble)
    exclude_geometry = [380.0, 420.0, 40.0, 40.0]

    zone3_path = session_container.add_zone(
        file_path=session_path,
        zone_index=3,
        zone_role="exclude",
        geometry_px=exclude_geometry,
        shape="circle",
    )

    print(f"  ✓ Exclude zone: {zone3_path}")
    print(f"    - Shape: circle")
    print(f"    - Purpose: exclude air bubble")

    # Add pixel-to-mm mapping
    pixel_to_mm_conversion = {
        "scale_x": holder_diameter_mm / holder_geometry[2],  # mm per pixel
        "scale_y": holder_diameter_mm / holder_geometry[3],
        "method": "holder_calibration",
    }

    mapping_path = session_container.add_image_mapping(
        file_path=session_path,
        sample_holder_zone_id="zone_001",
        pixel_to_mm_conversion=pixel_to_mm_conversion,
    )

    print(f"  ✓ Pixel-to-mm mapping: {mapping_path}")
    print(
        f"    - Scale: {pixel_to_mm_conversion['scale_x']:.4f} mm/px"
    )

    # Verify zones
    with h5py.File(session_path, "r") as f:
        assert "/images/zones/zone_001" in f
        assert "/images/zones/zone_002" in f
        assert "/images/zones/zone_003" in f

        zone1 = f["/images/zones/zone_001"]
        assert zone1.attrs["zone_role"] == "sample_holder"
        assert zone1.attrs["holder_diameter_mm"] == holder_diameter_mm

        zone2 = f["/images/zones/zone_002"]
        assert zone2.attrs["zone_role"] == "include"

        zone3 = f["/images/zones/zone_003"]
        assert zone3.attrs["zone_role"] == "exclude"

        print(f"  ✓ All zones verified in HDF5")

# ============================================================================
# Test 4: Add Measurement Points
# ============================================================================


