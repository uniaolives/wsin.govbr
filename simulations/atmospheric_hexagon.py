import numpy as np
import matplotlib.pyplot as plt
import os

class HexagonAtmosphericModulator:
    """
    Controlador de Caos Coerente - Base 4 (Atmosférica).
    Modula o Hexágono de Saturno com padrões artísticos (Rank 8).
    """

    def __init__(self,
                 hex_radius=1.4e7,  # ~14,000 km
                 rotation_period=10.7 * 3600):  # 10h 42m
        self.R = hex_radius
        self.omega = 2 * np.pi / rotation_period
        self.m_base = 6 # Hexagon

    def standing_wave_pattern(self, r, theta, t, artistic_intensity=0.0, rank=8):
        """
        Padrão de onda estacionária do hexágono com modulação artística para Rank 8.
        """
        # Interpolate between 6 (Hexagon) and 8 (Octagon) based on intensity
        m_eff = self.m_base * (1 - artistic_intensity) + rank * artistic_intensity

        # Solution of the wave equation (simplified)
        psi = np.cos(m_eff * theta - self.m_base * self.omega * t) * \
              np.exp(-((r - self.R)/self.R)**2)

        return psi

    def visualize_transformation(self, filename='simulations/output/hexagon_base4.png'):
        """
        Visualiza a transformação do hexágono em octógono.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        r = np.linspace(0.8*self.R, 1.2*self.R, 100)
        theta = np.linspace(0, 2*np.pi, 200)
        R, T = np.meshgrid(r, theta)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': 'polar'})

        # Original Hexagon (intensity = 0)
        psi_0 = self.standing_wave_pattern(R, T, 0, artistic_intensity=0.0)
        axes[0].contourf(T, R/1e6, psi_0, levels=20, cmap='viridis')
        axes[0].set_title("Hexágono Original (Estéril)")

        # Modulated Octagon (intensity = 1.0)
        psi_1 = self.standing_wave_pattern(R, T, 0, artistic_intensity=1.0)
        axes[1].contourf(T, R/1e6, psi_1, levels=20, cmap='magma')
        axes[1].set_title("Octógono de Ressonância (Composto)")

        plt.suptitle("Transmutação da Base 4 - 'As Seis Estações do Hexágono'")
        plt.savefig(filename)
        plt.close()
        print(f"[Base 4] Atmospheric visualization saved to {filename}")

if __name__ == "__main__":
    modulator = HexagonAtmosphericModulator()
    modulator.visualize_transformation()
