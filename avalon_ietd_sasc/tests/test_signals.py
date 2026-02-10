import pytest
from PyQt5.QtCore import QCoreApplication
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.system_manager import SystemManager
from database.db_manager import DatabaseManager

def test_system_manager_signals(qtbot):
    db = DatabaseManager(db_url="sqlite:///:memory:")
    db.initialize()
    manager = SystemManager(db_manager=db)

    with qtbot.waitSignal(manager.sensor_data_updated, timeout=2000) as blocker:
        manager.start()

    assert blocker.signal_triggered
    assert 'temperature' in blocker.args[0]

    manager.stop()
