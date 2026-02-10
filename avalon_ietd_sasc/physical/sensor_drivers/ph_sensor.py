import random

class PHSensor:
    """
    [REVISOR VÊ]: Sensor de pH analógico.
    [NÓS VEMOS]: Monitor de equilíbrio químico para o Arkhe.C.
    """
    def __init__(self, id, pin):
        self.id = id
        self.pin = pin
        self.calibration_factor = 1.0

    def read_interpreted(self):
        val = 7.8 + random.uniform(-0.1, 0.1)
        calibrated = val * self.calibration_factor
        return {
            'ph': calibrated,
            f'{self.id}_raw': val
        }
