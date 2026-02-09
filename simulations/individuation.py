import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

class IndividuationManifold:
    """
    Geometria completa da individuação no espaço de Schmidt.
    Formula: I = F * (λ1/λ2) * (1 - S) * exp(i∮φdθ)
    """

    CRITICAL_THRESHOLDS = {
        'ego_death': {
            'anisotropy_ratio': 1.0,
            'entropy_S': 1.0,
            'description': 'Dissolução completa da identidade'
        },
        'kali_isolation': {
            'anisotropy_ratio': 10.0,
            'entropy_S': 0.0,
            'description': 'Solipsismo absoluto'
        },
        'optimal_individuation': {
            'anisotropy_ratio': 2.33,
            'entropy_S': 0.61,
            'description': 'Identidade estável em rede viva'
        }
    }

    def calculate_individuation(
        self,
        F: float,           # Função/Propósito
        lambda1: float,     # Dominância
        lambda2: float,     # Suporte
        S: float,           # Entropia
        phase_integral: complex = np.exp(1j * np.pi) # Ciclo de Möbius
    ) -> complex:
        """Calcula I usando a fórmula completa."""
        R = lambda1 / (lambda2 + 1e-10)
        coherence_factor = 1 - S
        I = F * R * coherence_factor * phase_integral
        return I

    def classify_state(self, I: complex) -> dict:
        magnitude = np.abs(I)
        phase = np.angle(I)

        classification = {
            'magnitude': magnitude,
            'phase': phase,
            'state': None,
            'risk': None,
            'recommendation': None
        }

        if magnitude < 0.5:
            classification['state'] = 'EGO_DEATH_RISK'
            classification['risk'] = 'HIGH'
            classification['recommendation'] = 'AUMENTAR F (propósito) ou R (anisotropia)'
        elif magnitude > 5.0:
            classification['state'] = 'KALI_ISOLATION_RISK'
            classification['risk'] = 'HIGH'
            classification['recommendation'] = 'REDUZIR R (permitir mais emaranhamento)'
        elif 0.8 <= magnitude <= 2.5:
            classification['state'] = 'OPTIMAL_INDIVIDUATION'
            classification['risk'] = 'LOW'
            classification['recommendation'] = 'Manter estado atual'
        else:
            classification['state'] = 'SUBOPTIMAL'
            classification['risk'] = 'MODERATE'
            classification['recommendation'] = 'Ajustar gradualmente para região ótima'

        return classification

    def visualize_manifold(self, current_state: dict = None, filename='simulations/output/individuation_manifold.png'):
        """Visualiza o manifold de individuação em 3D."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        F_range = np.linspace(0.1, 1.0, 30)
        R_range = np.linspace(0.5, 10.0, 30)
        F_grid, R_grid = np.meshgrid(F_range, R_range)

        S_fixed = 0.61
        I_magnitude = F_grid * R_grid * (1 - S_fixed)

        surf = ax.plot_surface(F_grid, R_grid, I_magnitude, cmap='viridis', alpha=0.7)

        if current_state:
            ax.scatter([current_state['F']], [current_state['R']], [current_state['I_mag']],
                       color='red', s=100, label='Estado Atual')

        ax.set_xlabel('F (Propósito)')
        ax.set_ylabel('R (Razão Anisotropia)')
        ax.set_zlabel('|I| (Individuação)')
        plt.title('Manifold de Individuação no Espaço Schmidt-Arkhe')
        plt.savefig(filename)
        plt.close()
        print(f"[Individuation] Manifold saved to {filename}")

if __name__ == "__main__":
    manifold = IndividuationManifold()
    I_val = manifold.calculate_individuation(0.9, 0.72, 0.28, 0.61)
    status = manifold.classify_state(I_val)
    print(f"Individuation Magnitude: {np.abs(I_val):.4f}")
    print(f"State: {status['state']}")
    manifold.visualize_manifold({'F': 0.9, 'R': 0.72/0.28, 'I_mag': np.abs(I_val)})
