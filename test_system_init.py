import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent / "avalon_ietd_sasc"))

from main import EnvironmentalMonitoringSystem

def test_initialization():
    print("Testing system initialization...")
    try:
        # We need to mock sys.argv if we want to bypass argparse in a script
        sys.argv = ["main.py", "--quantum-mode"]

        system = EnvironmentalMonitoringSystem(quantum_mode=True)
        system.initialize()
        print("Initialization successful.")
        return True
    except Exception as e:
        print(f"Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Ensure we use offscreen for Qt
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    success = test_initialization()
    sys.exit(0 if success else 1)
