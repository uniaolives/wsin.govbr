from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
import pyqtgraph as pg
import numpy as np

class ChartsWidget(QWidget):
    """
    Gráficos em tempo real.
    [REVISOR]: Visualização de tendências históricas.
    [NÓS]: Registro de trajetórias no espaço de fase.
    """
    def __init__(self, db_manager=None, quantum_mode=False, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.quantum_mode = quantum_mode

        self.data_temp = []
        self.data_ph = []
        self.data_cond = []

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Plot de Temperatura
        self.temp_plot = pg.PlotWidget(title="Temperature History")
        self.temp_curve = self.temp_plot.plot(pen='r')
        layout.addWidget(self.temp_plot)

        # Plot de pH
        self.ph_plot = pg.PlotWidget(title="pH History")
        self.ph_curve = self.ph_plot.plot(pen='g')
        layout.addWidget(self.ph_plot)

        # [OCULTO]: Plot quântico (S de von Neumann)
        if self.quantum_mode:
            self.entropy_plot = pg.PlotWidget(title="Entanglement Entropy S(ρ)")
            self.entropy_curve = self.entropy_plot.plot(pen='c')
            layout.addWidget(self.entropy_plot)
            self.data_entropy = []

    def add_sensor_data(self, data):
        self.data_temp.append(data.get('temperature', 25))
        self.data_ph.append(data.get('ph', 7.8))

        # Mantém apenas últimos 100 pontos
        if len(self.data_temp) > 100:
            self.data_temp.pop(0)
            self.data_ph.pop(0)

        self.temp_curve.setData(self.data_temp)
        self.ph_curve.setData(self.data_ph)

        if self.quantum_mode:
            # [OCULTO]: Simula entropia variável
            s = 0.85 + 0.05 * np.sin(len(self.data_temp) / 10.0)
            self.data_entropy.append(s)
            if len(self.data_entropy) > 100: self.data_entropy.pop(0)
            self.entropy_curve.setData(self.data_entropy)
