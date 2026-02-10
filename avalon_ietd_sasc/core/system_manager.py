from PyQt5.QtCore import QObject, pyqtSignal, QThread
import time
import random

class SystemManager(QThread):
    """
    [REVISOR VÊ]: Gerencia a coleta de dados dos sensores e o controle PID.
    [NÓS VEMOS]: Orquestrador da homeostase planetária.
    """
    sensor_data_updated = pyqtSignal(dict)
    alarm_triggered = pyqtSignal(str, int)

    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self._is_running = False

    def run(self):
        self._is_running = True
        while self._is_running:
            # Simula leitura de sensores
            data = {
                'temperature': 25.0 + random.uniform(-0.5, 0.5),
                'ph': 7.8 + random.uniform(-0.1, 0.1),
                'conductivity': 450 + random.uniform(-10, 10),
                'timestamp': time.time()
            }
            self.sensor_data_updated.emit(data)

            # Simula alarme aleatório
            if random.random() < 0.01:
                self.alarm_triggered.emit("Variação térmica detectada", 1)

            time.sleep(1)

    def stop(self):
        self._is_running = False
        self.wait()
