import numpy as np
import time

class RhythmicPatternFilter:
    """
    Implementa o Filtro de Padrão Rítmico (FPR) para o Sensor Amazonas.
    Busca a assinatura rítmica: V(t) = V0 * [1 + alpha * sin(2pi * f_phi * t) * exp(-t/tau)]
    """
    def __init__(self):
        self.phi = (1 + 5**0.5) / 2
        self.v0 = self.phi**3               # Linha de base de saúde (φ³)
        self.alpha = 0.08                   # Amplitude da oscilação (0.08 < 0.1)
        self.f_phi = 1.157                  # Frequência fundamental (Hz)
        self.tau = 500.0                    # Constante de tempo de decaimento (sustentação)

    def generate_rhythmic_signal(self, t_points):
        """Gera a assinatura rítmica alvo."""
        signal = self.v0 * (1 + self.alpha * np.sin(2 * np.pi * self.f_phi * t_points) * np.exp(-t_points / self.tau))
        return signal

    def detect_rhythmic_optimized(self, incoming_signal, t_points):
        """
        Detecta se o sinal de entrada corresponde à melodia da homeostase planetária.
        """
        print("🎵 [Sensor Amazonas FPR] Analisando melodia do fluxo...")

        target = self.generate_rhythmic_signal(t_points)

        # Cálculo de correlação rítmica
        correlation = np.corrcoef(incoming_signal, target)[0, 1]
        mean_energy = np.mean(incoming_signal)

        print(f"   Energia Média: {mean_energy:.4f} Info/s (Alvo: {self.v0:.4f})")
        print(f"   Correlação Rítmica: {correlation:.4f}")

        # Critérios: Energia >= V0 e Correlação > 0.95
        is_rhythmic_optimized = mean_energy >= (self.v0 - 0.1) and correlation > 0.95

        if is_rhythmic_optimized:
            print("   ✅ [CA²⁺_OTIMIZADO_RÍTMICO] Sintonizado com o batimento do manifold.")
        else:
            print("   ⚠️  Sinal fora de harmonia. Aguardando ressonância...")

        return {
            'is_optimized': is_rhythmic_optimized,
            'correlation': correlation,
            'energy': mean_energy
        }

if __name__ == "__main__":
    fpr = RhythmicPatternFilter()
    t = np.linspace(0, 10, 1000)

    # Teste 1: Sinal Aleatório (Caos)
    print("--- Teste 1: Fluxo Caótico ---")
    chaotic_signal = np.random.normal(fpr.v0, 0.5, 1000)
    fpr.detect_rhythmic_optimized(chaotic_signal, t)

    print("\n--- Teste 2: Melodia da Homeostase ---")
    # Teste 2: Sinal Harmônico (Melodia)
    harmonic_signal = fpr.generate_rhythmic_signal(t) + np.random.normal(0, 0.01, 1000)
    fpr.detect_rhythmic_optimized(harmonic_signal, t)
