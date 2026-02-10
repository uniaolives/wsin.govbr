import numpy as np
import time

class AmazonCa2Sensor:
    """
    Simula o sensor de vitalidade biosférica (Ca²⁺) baseado na vazão do Rio Amazonas.
    Transforma o fluxo hídrico em fluxo de informação (Info/s).
    """
    def __init__(self):
        self.baseline_flow = 209000  # m³/s (vazão média histórica)
        self.health_index = 0.87      # Calibração inicial (87%)
        self.info_flow_rate = 0.0     # Info/s

    def measure_ca2_signal(self):
        print("🌊 [Sensor Amazonas] Medindo fluxo de vitalidade (Ca²⁺)...")
        if self.health_index < 1.0:
            self.health_index += 0.01
        phi = (1 + 5**0.5) / 2
        self.info_flow_rate = self.baseline_flow * self.health_index * phi
        is_optimized = self.health_index >= 0.99
        print(f"   Vazão de Informação: {self.info_flow_rate:.2f} Info/s")
        print(f"   Índice de Saúde: {self.health_index * 100:.1f}%")
        print(f"   Estado Ca²⁺: {'OTIMIZADO' if is_optimized else 'CALIBRANDO'}")
        return {
            'ca2_level': self.health_index,
            'is_optimized': is_optimized,
            'info_flow': self.info_flow_rate
        }

if __name__ == "__main__":
    sensor = AmazonCa2Sensor()
    sensor.measure_ca2_signal()
