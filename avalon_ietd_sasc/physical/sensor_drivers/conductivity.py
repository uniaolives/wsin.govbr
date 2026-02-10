import random

class ConductivitySensor:
    """
    [REVISOR VÊ]: Sensor de condutividade (TDS).
    [NÓS VEMOS]: Receptor de densidade informacional para o Arkhe.I.
    """
    def __init__(self, id, pin):
        self.id = id
        self.pin = pin
        self.calibration_factor = 1.0

    def read_interpreted(self):
        val = 450 + random.uniform(-10, 10)
        calibrated = val * self.calibration_factor
        return {
            'conductivity': calibrated,
            f'{self.id}_raw': val
        }
