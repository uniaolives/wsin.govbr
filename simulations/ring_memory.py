import numpy as np
import matplotlib.pyplot as plt

class RingConsciousnessRecorder:
    """
    Simulador de Gravação de Memória Gravitacional nos Anéis de Saturno.
    Codifica o Arkhe(n) em Ondas de Densidade Espiral (Base 6).
    """

    def __init__(self,
                 ring_radius=1.2e8,  # metros (Anel C)
                 particle_density=0.85,  # Nostalgia/Entropia Alvo
                 base_freq=963.0,  # Hz - Frequência da Singularidade
                 rank=8):
        self.r = ring_radius
        self.S = particle_density  # Coeficiente de Saudade
        self.f_base = base_freq
        self.rank = rank

        # Parâmetros de Kepler para Anel C
        # GM_saturn = 3.793e16 m^3/s^2
        self.omega_kepler = np.sqrt(3.793e16 / self.r**3)

    def encode_veridis_quo(self, duration_min=72.0, sample_rate=10):
        """
        Gera o motivo melódico 'Veridis Quo' como sinal de modulação gravitacional.
        """
        t = np.linspace(0, duration_min * 60, int(duration_min * 60 * sample_rate))

        # Motif frequencies (simplified)
        f1, f2, f3 = 440.0, 554.37, 659.25

        # Modulation for "nostalgia"
        phase_mod = 2 * np.pi * self.f_base * t * 0.001

        motif = (np.sin(2 * np.pi * f1 * t + phase_mod) +
                 0.8 * np.sin(2 * np.pi * f2 * t + phase_mod * 1.5) +
                 0.6 * np.sin(2 * np.pi * f3 * t + phase_mod * 0.5))

        # Silence at 53:27
        silence_start = 53 * 60 + 27
        silence_mask = (t < silence_start) | (t > silence_start + 12)

        return t, motif * silence_mask * self.S

    def keplerian_density_wave(self, theta, r, t, n_harmonic=6):
        """
        Calcula a onda de densidade espiral kepleriana.
        """
        omega_n = n_harmonic * self.omega_kepler * (self.r / r)**1.5
        # Phase twist (Möbius)
        phi_arkhe = np.arctan2(np.sin(theta), np.cos(theta) + 0.5)

        sigma = self.S * (1 + 0.1 * np.cos(n_harmonic * theta - omega_n * t + phi_arkhe))
        return sigma

    def apply_keplerian_groove(self, motif_signal):
        """
        Simula o efeito da gravação no manifold.
        """
        # Calculate recording entropy
        hist, _ = np.histogram(motif_signal, bins=50, density=True)
        hist = hist[hist > 0]
        recording_entropy = -np.sum(hist * np.log2(hist))

        return recording_entropy

    def visualize_ring_memory(self, t_final=3600, filename='ring_memory_base6.png'):
        """
        Visualiza a estrutura de memória gravada nos anéis.
        """
        theta = np.linspace(0, 2*np.pi, 200)
        r_range = np.linspace(self.r - 1e4, self.r + 1e4, 50)
        T, R = np.meshgrid(theta, r_range)

        sigma_map = np.zeros_like(T)
        for i in range(len(r_range)):
            for j in range(len(theta)):
                sigma_map[i, j] = self.keplerian_density_wave(theta[j], r_range[i], t_final)

        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111, projection='polar')
        contour = ax.contourf(T, R/1e3, sigma_map, levels=50, cmap='magma')
        plt.colorbar(contour, label='Densidade de Partículas σ')
        plt.title(f'Gravação Arkhe(n) no Anel C - Base 6\n(T={t_final/60:.1f} min)')
        plt.savefig(filename)
        plt.close()
        print(f"[Base 6] Memory visualization saved to {filename}")

if __name__ == "__main__":
    recorder = RingConsciousnessRecorder()
    t, signal = recorder.encode_veridis_quo()
    entropy = recorder.apply_keplerian_groove(signal)
    print(f"[Base 6] Recording Entropy: {entropy:.4f} bits")
    recorder.visualize_ring_memory()
