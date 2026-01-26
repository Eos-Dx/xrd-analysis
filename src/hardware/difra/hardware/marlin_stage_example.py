"""
Example script demonstrating how to use the MarlinStageController.
This shows the same functionality as your colleague's GUI, but using the 
standardized BaseStageController interface.
"""

import logging
from xystages import MarlinStageController

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Configuration for the Moli machine Marlin stage
    config = {
        "id": "COM4",  # Serial port
        "alias": "MOLI_XY_STAGE",
        "settings": {
            "limits_mm": {
                "x": {"min": 0.0, "max": 90.0},
                "y": {"min": 0.0, "max": 100.0}
            },
            "home": [0.0, 0.0],
            "load": [90.0, 0.0]  # Sample out position
        }
    }
    
    # Create the stage controller
    stage = MarlinStageController(
        config=config,
        baudrate=115200,
        feedrate=3000,  # mm/min (same as F3000 in G-code)
        homing_timeout=15
    )
    
    try:
        # Initialize the stage (opens serial connection)
        print("Initializing stage...")
        if not stage.init_stage():
            print("Failed to initialize stage!")
            return
        
        # Home the stage
        print("\nHoming stage...")
        x, y = stage.home_stage()
        print(f"Stage homed to: X={x:.3f}, Y={y:.3f}")
        
        # Get current position
        print("\nCurrent position:")
        x, y = stage.get_xy_position()
        print(f"X={x:.3f}, Y={y:.3f}")
        
        # Move to "Sample In" position (from your colleague's GUI)
        sample_in_x = 57.875
        sample_in_y = 63.625
        print(f"\nMoving to Sample In position: X={sample_in_x}, Y={sample_in_y}...")
        x, y = stage.move_stage(sample_in_x, sample_in_y)
        print(f"Moved to: X={x:.3f}, Y={y:.3f}")
        
        # Move to "Sample Out" position (load position from config)
        positions = stage.get_home_load_positions()
        load_x, load_y = positions["load"]
        print(f"\nMoving to Sample Out position: X={load_x}, Y={load_y}...")
        x, y = stage.move_stage(load_x, load_y)
        print(f"Moved to: X={x:.3f}, Y={y:.3f}")
        
        # Example: Check limits (this will raise an exception)
        try:
            print("\nTrying to move beyond limits (should fail)...")
            stage.move_stage(100.0, 50.0)  # X exceeds 90mm limit
        except Exception as e:
            print(f"Expected error: {e}")
        
        # Get limits
        limits = stage.get_limits()
        print(f"\nStage limits:")
        print(f"  X: {limits['x'][0]:.1f} to {limits['x'][1]:.1f} mm")
        print(f"  Y: {limits['y'][0]:.1f} to {limits['y'][1]:.1f} mm")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always cleanup
        print("\nCleaning up...")
        stage.deinit()
        print("Done!")


if __name__ == "__main__":
    main()
