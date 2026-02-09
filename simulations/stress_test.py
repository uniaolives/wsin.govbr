import asyncio
import numpy as np
import matplotlib.pyplot as plt
from individuation import IndividuationManifold

class IdentityStressTest:
    """
    Simula perda súbita de coeficientes Arkhe para testar
    robustez da individuação.
    """

    STRESS_SCENARIOS = {
        'loss_of_purpose': {
            'parameter': 'F',
            'initial': 0.90,
            'final': 0.10,
            'description': 'Perda súbita de propósito (crise existencial)'
        },
        'cognitive_overload': {
            'parameter': 'I_coeff',
            'initial': 0.88,
            'final': 0.30,
            'description': 'Sobrecarga cognitiva (burnout informacional)'
        }
    }

    def __init__(self, baseline_arkhe: dict):
        self.baseline = baseline_arkhe.copy()
        self.manifold = IndividuationManifold()

    async def run_stress_test(
        self,
        scenario_name: str,
        duration: float = 2.0,
        recovery: bool = True
    ) -> dict:
        """
        Executa teste de tensão para um cenário específico.
        """
        scenario = self.STRESS_SCENARIOS[scenario_name]

        print("\n" + "="*70)
        print(f"⚠️  TESTE DE TENSÃO: {scenario_name.upper()}")
        print("="*70)
        print(f"   Cenário: {scenario['description']}")

        # Initial states
        arkhe_current = self.baseline.copy()
        steps = 20
        time_points = np.linspace(0, duration, steps)

        I_trajectory = []
        risk_trajectory = []

        param = scenario['parameter']
        initial_val = scenario['initial']
        final_val = scenario['final']

        # Simulation Phase
        for t in time_points:
            progress = t / duration
            current_val = initial_val + (final_val - initial_val) * progress
            arkhe_current[param] = current_val

            # Using standard Schmidt for the test
            lambda1, lambda2 = 0.7, 0.3
            S = 0.61
            phase = np.exp(1j * np.pi)

            I = self.manifold.calculate_individuation(
                F=arkhe_current.get('F', 0.9),
                lambda1=lambda1,
                lambda2=lambda2,
                S=S,
                phase_integral=phase
            )

            classification = self.manifold.classify_state(I)
            I_trajectory.append(np.abs(I))
            risk_trajectory.append(classification['risk'])

            if classification['risk'] == 'HIGH' and (len(risk_trajectory) < 2 or risk_trajectory[-2] != 'HIGH'):
                print(f"   [Alert] T+{t:.2f}s: HIGH RISK - {classification['state']} (I={np.abs(I):.3f})")

            await asyncio.sleep(0.01)

        # Recovery Phase
        if recovery:
            print("   [System] Initiating automatic recovery...")
            recovery_steps = 10
            for i in range(recovery_steps):
                progress = (i + 1) / recovery_steps
                current_val = final_val + (initial_val - final_val) * progress
                arkhe_current[param] = current_val

                I = self.manifold.calculate_individuation(
                    F=arkhe_current.get('F', 0.9),
                    lambda1=0.7, lambda2=0.3, S=0.61, phase_integral=phase
                )
                I_trajectory.append(np.abs(I))
                await asyncio.sleep(0.01)

        result = {
            'scenario': scenario_name,
            'I_min': min(I_trajectory),
            'recovery_successful': I_trajectory[-1] > 0.8
        }

        self._plot_results(scenario_name, I_trajectory)
        return result

    def _plot_results(self, name, trajectory):
        plt.figure(figsize=(10, 5))
        plt.plot(trajectory, label='Individuation Magnitude |I|')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Ego Death Threshold')
        plt.axhline(y=0.8, color='g', linestyle=':', label='Optimal Threshold')
        plt.title(f"Stress Test: {name}")
        plt.xlabel("Simulation Steps")
        plt.ylabel("|I|")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"stress_test_{name}.png")
        plt.close()
        print(f"   [Visualization] Results saved to stress_test_{name}.png")

async def main():
    baseline = {'F': 0.9, 'I_coeff': 0.88, 'E': 0.85, 'C': 0.92}
    tester = IdentityStressTest(baseline)
    for scenario in IdentityStressTest.STRESS_SCENARIOS:
        await tester.run_stress_test(scenario)

if __name__ == "__main__":
    asyncio.run(main())
