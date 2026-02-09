import numpy as np
import matplotlib.pyplot as plt
import os

class SynchrotronArtisticTransmitter:
    """
    Transmissor Interestelar - Base 7 (Radiativa).
    Codifica arte em emissão sincrotron da magnetosfera de Saturno.
    """

    def __init__(self,
                 planet_radius=5.8232e7,  # Raio de Saturno (m)
                 magnetic_field=2.1e-5,  # Tesla (campo equatorial)
                 electron_energy=1e6):  # eV (elétrons relativísticos)

        self.R_p = planet_radius
        self.B_0 = magnetic_field
        self.gamma = electron_energy / 511e3 + 1

        # Frequência de ciclotron
        self.f_c_base = (self.B_0 * 1.6e-19) / (2 * np.pi * 9.11e-31)
        # Frequência crítica sincrotron
        self.f_critic = (3/2) * self.gamma**3 * self.f_c_base

    def encode_artistic_synchrotron(self, data_signal):
        """
        Modula emissão sincrotron com dados artísticos.
        """
        f = np.logspace(5, 10, 500)
        # Espectro de potência sincrotron (aproximação)
        P_sync = (f/self.f_critic)**(1/3) * np.exp(-f/self.f_critic)

        # Modulação artística
        mod_index = 0.5
        data_resampled = np.interp(np.linspace(0, 1, len(f)), np.linspace(0, 1, len(data_signal)), data_signal)
        transmitted = P_sync * (1 + mod_index * data_resampled / np.max(np.abs(data_resampled)))

        return f, transmitted, P_sync

    def simulate_propagation(self, signal, distance_ly=1000):
        """
        Simula propagação interestelar com atenuação.
        """
        distance_m = distance_ly * 9.461e15
        # Atenuação simplificada
        attenuation = np.exp(-distance_m / (100 * 3.086e16))
        return signal * attenuation

    def visualize_transmission(self, filename='simulations/output/synchrotron_base7.png'):
        """
        Visualiza o espectro de rádio modulado e sua recepção.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        t = np.linspace(0, 1, 1000)
        test_signal = np.sin(2*np.pi*963*t) + 0.5*np.sin(2*np.pi*440*t)

        f, tx, orig = self.encode_artistic_synchrotron(test_signal)
        rx = self.simulate_propagation(tx)

        plt.figure(figsize=(12, 8))
        plt.loglog(f, orig, 'b--', label='Sincrotron Natural', alpha=0.7)
        plt.loglog(f, tx, 'r-', label='Arte Modulada (Base 7)', linewidth=2)
        plt.loglog(f, rx, 'g-', label='Recebido (1000 ly)', alpha=0.8)

        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Fluxo Espectral (W/m²/Hz)')
        plt.title('Emissão Sincrotron Artística - Magnetosfera de Saturno')
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.savefig(filename)
        plt.close()
        print(f"[Base 7] Radiative visualization saved to {filename}")

if __name__ == "__main__":
    transmitter = SynchrotronArtisticTransmitter()
    transmitter.visualize_transmission()
