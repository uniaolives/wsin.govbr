from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFormLayout

class SettingsDialog(QDialog):
    """
    Diálogo de configurações.
    [REVISOR]: Configuração de rede e limites de alarme.
    [NÓS]: Definição de constantes universais.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System Settings")
        self.setMinimumWidth(400)

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self.ip_input = QLineEdit("192.168.1.100")
        form.addRow("Arduino IP:", self.ip_input)

        self.temp_min = QLineEdit("24.0")
        form.addRow("Min Temp (°C):", self.temp_min)

        self.temp_max = QLineEdit("27.0")
        form.addRow("Max Temp (°C):", self.temp_max)

        # [OCULTO]: Campo quântico
        self.phi_const = QLineEdit("1.618033")
        self.phi_const.setReadOnly(True)
        form.addRow("Golden Ratio (φ):", self.phi_const)

        layout.addLayout(form)

        btns = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        btns.addStretch()
        btns.addWidget(save_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)
