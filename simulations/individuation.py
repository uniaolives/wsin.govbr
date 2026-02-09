import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

class IndividuationManifold:
    """
    Geometria completa da individuação no espaço de Schmidt.
    Formalização: I = F * (λ1/λ2) * (1 - S) * e^(i∮φdθ)
    """

    CRITICAL_THRESHOLDS = {
        'ego_death': {
            'anisotropy_ratio': 1.0,  # λ₁/λ₂ → 1 (fusão total)
            'entropy_S': 1.0,          # S → log(2) (entropia máxima)
            'description': 'Dissolução completa da identidade'
        },
        'kali_isolation': {
            'anisotropy_ratio': 10.0,  # λ₁/λ₂ → ∞ (separação total)
            'entropy_S': 0.0,           # S → 0 (sem emaranhamento)
            'description': 'Solipsismo absoluto'
        },
        'optimal_individuation': {
            'anisotropy_ratio': 2.33,  # λ₁/λ₂ = 0.7/0.3
            'entropy_S': 0.61,          # S(0.7, 0.3)
            'description': 'Identidade estável em rede viva'
        }
    }

    def __init__(self):
        self.simplex = None
        self.attractor = None

    def calculate_individuation(
        self,
        F: float,           # Função/Propósito
        lambda1: float,     # Dominância
        lambda2: float,     # Suporte
        S: float,           # Entropia
        phase_integral: complex  # Ciclo de Möbius
    ) -> complex:
        """
        Calcula I usando a fórmula completa.
        I = F · (λ₁/λ₂) · (1 - S) · e^(i∮φdθ)
        """
        # Razão de anisotropia
        R = lambda1 / lambda2 if lambda2 > 0 else 100.0

        # Fator de coerência
        coherence_factor = 1 - S

        # Individuação complexa (tem magnitude e fase)
        I = F * R * coherence_factor * phase_integral

        return I

    def classify_state(self, I: complex) -> dict:
        """
        Classifica o estado de individuação baseado em I.
        """
        magnitude = np.abs(I)
        phase = np.angle(I)

        classification = {
            'magnitude': magnitude,
            'phase': phase,
            'state': None,
            'risk': None,
            'recommendation': None
        }

        # Classifica baseado na magnitude
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

        # Classifica baseado na fase
        phase_normalized = phase % (2 * np.pi)

        if np.abs(phase_normalized - np.pi) < 0.1:
            classification['moebius_completion'] = 'COMPLETE'
        else:
            classification['moebius_completion'] = 'INCOMPLETE'
            classification['phase_error'] = phase_normalized - np.pi

        return classification

    def visualize_individuation_manifold(self, current_state: dict = None, filename='individuation_manifold.png'):
        """
        Visualiza o manifold de individuação em 3D.
        """
        try:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Cria grade de parâmetros
            F_range = np.linspace(0.1, 1.0, 30)
            R_range = np.linspace(0.5, 10.0, 30) # Increased R range to see isolation

            F_grid, R_grid = np.meshgrid(F_range, R_range)

            # Calcula individuação para cada ponto (assumindo S fixo)
            S_fixed = 0.61  # Entropia ótima
            phase_fixed = np.exp(1j * np.pi)  # Möbius completo

            I_magnitude = F_grid * R_grid * (1 - S_fixed)

            # Plota superfície
            surf = ax.plot_surface(
                F_grid, R_grid, I_magnitude,
                cmap='viridis',
                alpha=0.7,
                edgecolor='none'
            )

            # Helper to add safe contours
            def safe_contour(levels, colors, linestyles='-'):
                actual_levels = [l for l in levels if np.min(I_magnitude) < l < np.max(I_magnitude)]
                if actual_levels:
                    ax.contour(F_grid, R_grid, I_magnitude, levels=actual_levels, colors=colors, linewidths=2, linestyles=linestyles)

            safe_contour([0.5], 'red', '--')
            safe_contour([5.0], 'orange', '--')
            safe_contour([0.8, 2.5], 'green', '-')

            # Estado atual (se fornecido)
            if current_state:
                ax.scatter(
                    [current_state['F']],
                    [current_state['R']],
                    [current_state['I_magnitude']],
                    color='blue',
                    s=200,
                    marker='o',
                    label='Estado Atual'
                )

            ax.set_xlabel('F (Função/Propósito)')
            ax.set_ylabel('R (Razão λ₁/λ₂)')
            ax.set_zlabel('|I| (Magnitude da Individuação)')
            ax.set_title('Manifold de Individuação no Espaço Schmidt-Arkhe')

            # Adiciona legenda
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='red', linestyle='--', lw=2, label='Ego Death (I < 0.5)'),
                Line2D([0], [0], color='green', lw=2, label='Região Ótima (0.8 < I < 2.5)'),
                Line2D([0], [0], color='orange', linestyle='--', lw=2, label='Kali Isolation (I > 5.0)')
            ]

            if current_state:
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                          markersize=10, label='Estado Atual')
                )

            ax.legend(handles=legend_elements, loc='upper left')

            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"[Visualization] Manifold saved to {filename}")
            plt.close(fig)
            return True
        except Exception as e:
            print(f"[Visualization] Error generating manifold plot: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    manifold = IndividuationManifold()
    print("🧮 Geometria da Individuação carregada")

    # Calculate optimal point
    I_opt = manifold.calculate_individuation(
        F=0.9,
        lambda1=0.7,
        lambda2=0.3,
        S=0.61,
        phase_integral=np.exp(1j * np.pi)
    )
    classification = manifold.classify_state(I_opt)
    print(f"Optimal I: {I_opt}")
    print(f"Classification: {classification['state']} (Magnitude: {classification['magnitude']:.4f})")

    # Generate visualization
    manifold.visualize_individuation_manifold({
        'F': 0.9,
        'R': 0.7/0.3,
        'I_magnitude': np.abs(I_opt)
    })
