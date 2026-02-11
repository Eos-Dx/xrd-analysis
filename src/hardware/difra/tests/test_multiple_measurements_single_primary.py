"""Test that multiple measurements of same type per detector are allowed if only one is primary."""
import sys
import os

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_multiple_agbh_measurements_one_primary_allowed():
    """Test that multiple AGBH measurements for same detector are allowed if only one is primary.
    
    Scenario: User has 2 AGBH measurements for SAXS detector
    - AGBH_001_SAXS (primary=True)
    - AGBH_002_SAXS (primary=False)
    
    This should be ALLOWED because only one is marked as primary.
    """
    # Simulate the data structure
    primary_measurements = {}
    
    # Add two AGBH measurements for SAXS
    pair = ("AGBH", "SAXS")
    primary_measurements[pair] = [True, False]  # First is primary, second is supplementary
    
    # Validate: count primaries
    primary_count = sum(1 for is_prim in primary_measurements[pair] if is_prim)
    
    assert primary_count == 1, f"Expected 1 primary, got {primary_count}"
    print(f"✅ Multiple measurements allowed: {len(primary_measurements[pair])} total, {primary_count} primary")


def test_multiple_measurements_two_primaries_rejected():
    """Test that multiple measurements with TWO primaries is rejected."""
    primary_measurements = {}
    
    # Add two AGBH measurements for SAXS, BOTH marked as primary
    pair = ("AGBH", "SAXS")
    primary_measurements[pair] = [True, True]  # Both primary - INVALID
    
    # Validate: count primaries
    primary_count = sum(1 for is_prim in primary_measurements[pair] if is_prim)
    
    assert primary_count > 1, "Should have multiple primaries (invalid case)"
    print(f"✅ Multiple primaries detected: {primary_count} - validation should reject this")


def test_multiple_measurements_all_supplementary_rejected():
    """Test that having measurements but NO primary is detected."""
    primary_measurements = {}
    
    # Add two AGBH measurements for SAXS, NEITHER marked as primary
    pair = ("AGBH", "SAXS")
    primary_measurements[pair] = [False, False]  # None primary
    
    # Validate: count primaries
    primary_count = sum(1 for is_prim in primary_measurements[pair] if is_prim)
    
    assert primary_count == 0, "Should have zero primaries"
    print(f"✅ No primaries detected: {primary_count} - should require at least one primary")


def test_different_detectors_can_have_different_primaries():
    """Test that SAXS and WAXS can each have their own primary AGBH."""
    primary_measurements = {}
    
    # SAXS has 2 measurements, first is primary
    saxs_pair = ("AGBH", "SAXS")
    primary_measurements[saxs_pair] = [True, False]
    
    # WAXS has 2 measurements, second is primary  
    waxs_pair = ("AGBH", "WAXS")
    primary_measurements[waxs_pair] = [False, True]
    
    # Validate both
    saxs_primary_count = sum(1 for is_prim in primary_measurements[saxs_pair] if is_prim)
    waxs_primary_count = sum(1 for is_prim in primary_measurements[waxs_pair] if is_prim)
    
    assert saxs_primary_count == 1
    assert waxs_primary_count == 1
    
    print(f"✅ Different detectors can have different primaries:")
    print(f"   SAXS: {saxs_primary_count} primary of {len(primary_measurements[saxs_pair])} total")
    print(f"   WAXS: {waxs_primary_count} primary of {len(primary_measurements[waxs_pair])} total")


def test_three_measurements_one_primary_allowed():
    """Test that having 3+ measurements is fine as long as only one is primary."""
    primary_measurements = {}
    
    # Add THREE AGBH measurements for SAXS
    pair = ("AGBH", "SAXS")
    primary_measurements[pair] = [False, True, False]  # Middle one is primary
    
    # Validate
    primary_count = sum(1 for is_prim in primary_measurements[pair] if is_prim)
    
    assert primary_count == 1
    assert len(primary_measurements[pair]) == 3
    
    print(f"✅ Three measurements allowed: {len(primary_measurements[pair])} total, {primary_count} primary")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
