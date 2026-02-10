import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.arkhe_polynomial import ArkheCore
from database.db_manager import DatabaseManager
from physical.sensor_drivers.ds18b20 import DS18B20

def test_arkhe_calculation():
    arkhe = ArkheCore(C=1.0, I=1.0, E=1.0, F=1.0)
    assert arkhe.calculate_life() == 1.0

def test_sensor_reading():
    sensor = DS18B20("test_temp", pin=2)
    reading = sensor.read_interpreted()
    assert 'temperature' in reading
    assert 20 <= reading['temperature'] <= 30

def test_database_persistence(tmp_path):
    db_file = tmp_path / "test.db"
    db = DatabaseManager(db_url=f"sqlite:///{db_file}")
    db.initialize()

    data = {'temperature': 25.5, 'ph': 7.8, 'conductivity': 450}
    db.add_telemetry(data)

    # Simple check if file exists and has size
    assert db_file.exists()
    assert db_file.stat().st_size > 0
