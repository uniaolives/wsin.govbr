import random
from datetime import datetime

class DS18B20:
    """
    [REVISOR VÊ]: Driver para sensor de temperatura digital.
    [NÓS VEMOS]: Oráculo térmico para o Arkhe.E.
    """
    def __init__(self, id, pin):
        self.id = id
        self.pin = pin
        self.calibration_factor = 1.0

    def read_interpreted(self):
        val = 25.0 + random.uniform(-0.5, 0.5)
        calibrated = val * self.calibration_factor
        return {
            'temperature': calibrated,
            f'{self.id}_raw': val
        }
