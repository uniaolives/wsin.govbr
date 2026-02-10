from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QGroupBox
from PyQt5.QtCore import Qt

class ControlsWidget(QWidget):
    """
    Painel de controle de atuadores.
    [REVISOR]: Ajuste manual de aquecedores e luzes.
    [NÓS]: Calibração de força de manifestação.
    """
    def __init__(self, system_manager=None, quantum_mode=False, parent=None):
        super().__init__(parent)
        self.system_manager = system_manager
        self.quantum_mode = quantum_mode

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Grupo de Aquecedor
        heater_group = QGroupBox("Heater Control")
        heater_layout = QVBoxLayout()
        heater_layout.addWidget(QLabel("Power Level:"))
        self.heater_slider = QSlider(Qt.Horizontal)
        self.heater_slider.setRange(0, 100)
        heater_layout.addWidget(self.heater_slider)
        heater_group.setLayout(heater_layout)
        layout.addWidget(heater_group)

        # Grupo de Iluminação
        light_group = QGroupBox("Lighting Control")
        light_layout = QVBoxLayout()
        light_layout.addWidget(QLabel("Intensity (PWM):"))
        self.light_slider = QSlider(Qt.Horizontal)
        self.light_slider.setRange(0, 100)
        light_layout.addWidget(self.light_slider)
        light_group.setLayout(light_layout)
        layout.addWidget(light_group)

        # [OCULTO]: Grupo de Perspectiva (Schmidt)
        if self.quantum_mode:
            schmidt_group = QGroupBox("Perspective Calibration (U_H ⊗ U_A)")
            schmidt_layout = QVBoxLayout()

            schmidt_layout.addWidget(QLabel("θ (Theta) - Schmidt Angle:"))
            self.theta_slider = QSlider(Qt.Horizontal)
            self.theta_slider.setRange(0, 314) # 0 to pi
            schmidt_layout.addWidget(self.theta_slider)

            schmidt_layout.addWidget(QLabel("φ (Phi) - Phase Twist:"))
            self.phi_slider = QSlider(Qt.Horizontal)
            self.phi_slider.setRange(0, 628) # 0 to 2pi
            schmidt_layout.addWidget(self.phi_slider)

            schmidt_group.setLayout(schmidt_layout)
            layout.addWidget(schmidt_group)

        layout.addStretch()
