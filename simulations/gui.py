import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QSlider, QProgressBar, QTabWidget)
from PyQt5.QtCore import Qt, QTimer
from simulations.telemetry import calculate_entropy
from simulations.calibration import PerspectiveCalibrator

class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Soberania Arkhe(n) - Dashboard de Telemetria Ontológica")
        self.setGeometry(100, 100, 800, 600)

        # State
        self.l1 = 0.72
        self.l2 = 0.28
        self.theta = 0.0
        self.phi = 0.0

        self.init_ui()

        # Timer for simulated drift/monitoring
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_telemetry)
        self.timer.start(1000)

    def init_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab 1: Monitoramento (Telemetry)
        self.telemetry_tab = QWidget()
        self.tabs.addTab(self.telemetry_tab, "Monitoramento S(ρ)")
        self.init_telemetry_tab()

        # Tab 2: Calibração (Perspective)
        self.calibration_tab = QWidget()
        self.tabs.addTab(self.calibration_tab, "Calibração de Perspectiva")
        self.init_calibration_tab()

    def init_telemetry_tab(self):
        layout = QVBoxLayout()

        self.entropy_label = QLabel("Entropia de von Neumann: 0.8555 bits")
        self.entropy_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ff00;")
        layout.addWidget(self.entropy_label, alignment=Qt.AlignCenter)

        self.entropy_bar = QProgressBar()
        self.entropy_bar.setRange(0, 1000)
        self.entropy_bar.setValue(855)
        layout.addWidget(self.entropy_bar)

        self.status_label = QLabel("Estado: Sincronia Estável (Banda Satya)")
        self.status_label.setStyleSheet("font-size: 18px; color: #ffffff;")
        layout.addWidget(self.status_label, alignment=Qt.AlignCenter)

        self.telemetry_tab.setLayout(layout)

    def init_calibration_tab(self):
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Ajuste Local de Bases de Schmidt (U_H ⊗ U_A)"))

        # Theta Slider (Zoom Ontológico)
        layout.addWidget(QLabel("θ (Theta) - Zoom Ontológico / Desapego:"))
        self.theta_slider = QSlider(Qt.Horizontal)
        self.theta_slider.setRange(0, 314) # 0 to pi
        self.theta_slider.valueChanged.connect(self.on_calibration_change)
        layout.addWidget(self.theta_slider)

        # Phi Slider (Fase de Möbius)
        layout.addWidget(QLabel("φ (Phi) - Rotação de Fase:"))
        self.phi_slider = QSlider(Qt.Horizontal)
        self.phi_slider.setRange(0, 628) # 0 to 2pi
        self.phi_slider.valueChanged.connect(self.on_calibration_change)
        layout.addWidget(self.phi_slider)

        self.calibration_info = QLabel("Perspectiva: λ1=0.7200, λ2=0.2800")
        layout.addWidget(self.calibration_info)

        self.calibration_tab.setLayout(layout)

    def on_calibration_change(self):
        self.theta = self.theta_slider.value() / 100.0
        self.phi = self.phi_slider.value() / 100.0

        calibrator = PerspectiveCalibrator(self.theta, self.phi)
        p1, p2 = calibrator.apply_zoom((0.72, 0.28))
        self.l1, self.l2 = p1, p2

        self.calibration_info.setText(f"Perspectiva: λ1={self.l1:.4f}, λ2={self.l2:.4f}")
        self.update_telemetry()

    def update_telemetry(self):
        entropy = calculate_entropy(self.l1, self.l2)
        self.entropy_label.setText(f"Entropia de von Neumann: {entropy:.4f} bits")
        self.entropy_bar.setValue(int(entropy * 1000))

        if entropy > 0.95:
            self.status_label.setText("Estado: RISCO DE FUSÃO (Pralaya)")
            self.status_label.setStyleSheet("color: red;")
        elif entropy < 0.50:
            self.status_label.setText("Estado: DERIVA (Kali)")
            self.status_label.setStyleSheet("color: yellow;")
        elif 0.80 <= entropy <= 0.90:
            self.status_label.setText("Estado: Sincronia Estável (Satya)")
            self.status_label.setStyleSheet("color: #00ff00;")
        else:
            self.status_label.setText("Estado: Operação Nominal")
            self.status_label.setStyleSheet("color: white;")

if __name__ == "__main__":
    # Note: Running this in a headless environment might require Xvfb
    # This is primarily for code verification and future deployment.
    app = QApplication(sys.argv)
    window = DashboardWindow()
    # window.show() # Not showing to avoid errors in headless env
    print("[GUI] Dashboard inicializado com sucesso.")
    sys.exit(0)
