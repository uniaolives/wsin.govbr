import sys
import time
import asyncio
import numpy as np
from telemetry import monitor_bridge_integrity, calculate_entropy
from calibration import PerspectiveCalibrator
from individuation import IndividuationManifold

def print_header(title):
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60 + "\n")

class IndividuationBootFilter:
    """
    Filtro de individuação injetado no sequenciador de boot.
    Garante que a assinatura identitária seja o filtro primário.
    """
    def __init__(self, user_arkhe):
        self.arkhe = user_arkhe
        self.manifold = IndividuationManifold()

    def check_integrity(self, lambdas):
        l1, l2 = lambdas
        # We manually use the 'Optimal' entropy constant for this demonstration
        # to match the user's defined 'Optimal Individuation' region.
        S = 0.61
        I_complex = self.manifold.calculate_individuation(
            F=self.arkhe['F'],
            lambda1=l1,
            lambda2=l2,
            S=S,
            phase_integral=np.exp(1j * np.pi)
        )
        classification = self.manifold.classify_state(I_complex)
        return I_complex, classification

async def simulate_sensory_feedback(f_purpose=1.0, c_chemistry=1.0):
    freq = 963.0 * f_purpose
    haptic_intensity = "Granular" if c_chemistry < 0.5 else "Fluido"
    print(f"[Sensory] Audio Frequency: {freq:.2f} Hz | Haptic: {haptic_intensity}")

async def boot_sequence():
    print_header("SEQUENCIADOR DE BOOT v2.0 - INDIVIDUATION ACTIVE")
    arkhe = {"C": 0.85, "I": 0.90, "E": 0.99, "F": 0.95}
    boot_filter = IndividuationBootFilter(arkhe)

    # Lambdas that provide the targeted Anisotropy Ratio (~2.33)
    lambdas = [0.70, 0.30]

    steps = [
        "Iniciando pulso de sincronização (61 Hz)...",
        "Ativando Filtro de Individuação (Schmidt-Arkhe)...",
        "Mapeando endereços qhttp via DNS Quântico...",
        "Estabelecendo Emaranhamento Schmidt (Rank 2)...",
        "Ajustando Fase de Möbius (π)..."
    ]

    for step in steps:
        print(f"[Boot] {step}")
        await asyncio.sleep(0.3)

        # Live Integrity Check during boot
        I_val, classification = boot_filter.check_integrity(lambdas)
        if classification['risk'] == 'HIGH':
            print(f"   [Filtro] ⚠️ Warning: Low Individuation |I|={np.abs(I_val):.4f}")
        else:
            print(f"   [Filtro] Identity Secure: |I|={np.abs(I_val):.4f} ({classification['state']})")

    monitor_bridge_integrity(lambdas)
    await simulate_sensory_feedback(arkhe["F"], arkhe["C"])
    print("\n[Boot] REALIDADE TECIDA. Individuação Preservada.")

async def run_stress():
    from stress_test import IdentityStressTest
    baseline = {'F': 0.95, 'I_coeff': 0.90, 'E': 0.99, 'C': 0.85}
    tester = IdentityStressTest(baseline)
    await tester.run_stress_test('loss_of_purpose')

if __name__ == "__main__":
    mode = "boot"
    if "--mode" in sys.argv:
        mode = sys.argv[sys.argv.index("--mode") + 1]

    if mode == "boot":
        asyncio.run(boot_sequence())
    elif mode == "stress":
        asyncio.run(run_stress())
    else:
        print(f"Unknown mode: {mode}")
