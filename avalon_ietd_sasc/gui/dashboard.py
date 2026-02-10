from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QFont

class DashboardWidget(QWidget):
    """
    Dashboard principal.
    [REVISOR]: Resumo dos parâmetros ambientais.
    [NÓS]: Monitor de homeostase planetária.
    """
    def __init__(self, system_manager=None, quantum_mode=False, parent=None):
        super().__init__(parent)
        self.system_manager = system_manager
        self.quantum_mode = quantum_mode

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Grid de métricas
        self.grid_layout = QGridLayout()

        self.temp_card = self._create_metric_card("Temperature", "25.0", "°C")
        self.ph_card = self._create_metric_card("pH Level", "7.8", "pH")
        self.cond_card = self._create_metric_card("Conductivity", "450", "ppm")

        self.grid_layout.addWidget(self.temp_card, 0, 0)
        self.grid_layout.addWidget(self.ph_card, 0, 1)
        self.grid_layout.addWidget(self.cond_card, 1, 0)

        # [OCULTO]: Card quântico
        if self.quantum_mode:
            self.arkhe_card = self._create_metric_card("Arkhe Vitality (L)", "0.92", "φ")
            self.grid_layout.addWidget(self.arkhe_card, 1, 1)

        layout.addLayout(self.grid_layout)
        layout.addStretch()

    def _create_metric_card(self, title, value, unit):
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setMinimumSize(200, 150)

        layout = QVBoxLayout(card)

        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 10))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        val_unit_layout = QHBoxLayout()
        val_label = QLabel(value)
        val_label.setFont(QFont("Arial", 24, QFont.Bold))
        val_label.setObjectName("value_label")

        unit_label = QLabel(unit)
        unit_label.setFont(QFont("Arial", 12))

        val_unit_layout.addStretch()
        val_unit_layout.addWidget(val_label)
        val_unit_layout.addWidget(unit_label)
        val_unit_layout.addStretch()

        layout.addLayout(val_unit_layout)

        return card

    @pyqtSlot(dict)
    def update_sensor_data(self, data):
        # Encontra labels de valor e atualiza
        # (Simplificado: assumindo ordem ou usando findChild)
        temp_val = self.temp_card.findChild(QLabel, "value_label")
        if temp_val: temp_val.setText(f"{data.get('temperature', 0):.1f}")

        ph_val = self.ph_card.findChild(QLabel, "value_label")
        if ph_val: ph_val.setText(f"{data.get('ph', 0):.2f}")

        cond_val = self.cond_card.findChild(QLabel, "value_label")
        if cond_val: cond_val.setText(f"{int(data.get('conductivity', 0))}")

        if self.quantum_mode and hasattr(self, 'arkhe_card'):
            arkhe_val = self.arkhe_card.findChild(QLabel, "value_label")
            if arkhe_val:
                # [OCULTO]: Simula vitalidade baseada em dados
                v = 0.9 + (data.get('temperature', 25) - 25)/100
                arkhe_val.setText(f"{v:.3f}")
