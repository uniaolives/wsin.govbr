import numpy as np
import matplotlib.pyplot as plt
import os

class SynchrotronArtisticTransmitter:
    """
    Transmissor Interestelar - Base 7 (Radiativa).
    Codifica arte em emissão sincrotron da magnetosfera de Saturno.
    """

    def __init__(self,
                 magnetic_field=2.1e-5,  # Tesla
                 electron_energy=1e6):  # eV
        self.B_0 = magnetic_field
        self.gamma = electron_energy / 511e3 + 1
        # Critical frequency
        self.f_c = (3/2) * self.gamma**3 * (self.B_0 * 1.6e-19 / (2 * np.pi * 9.11e-31))

    def encode_signal(self, data_signal):
        """
        Modula emissão sincrotron com dados artísticos.
        """
        f = np.logspace(5, 10, 500)
        # Sincrotron power spectrum (approximation)
        P_sync = (f/self.f_c)**(1/3) * np.exp(-f/self.f_c)

        # Add artistic modulation
        mod_index = 0.5
        # Resample or truncate data_signal to match frequency points
        data_resampled = np.interp(np.linspace(0, 1, len(f)), np.linspace(0, 1, len(data_signal)), data_signal)
        transmitted = P_sync * (1 + mod_index * data_resampled / (np.max(np.abs(data_resampled)) + 1e-10))

        return f, transmitted, P_sync

    def visualize_transmission(self, filename='simulations/output/synchrotron_base7.png'):
        """
        Visualiza o espectro de rádio modulado.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        t = np.linspace(0, 1, 1000)
        test_signal = np.sin(2*np.pi*963*t) + 0.5*np.sin(2*np.pi*440*t)

        f, tx, orig = self.encode_signal(test_signal)

        plt.figure(figsize=(10, 6))
        plt.loglog(f, orig, 'b--', label='Sincrotron Natural', alpha=0.7)
        plt.loglog(f, tx, 'r-', label='Arte Modulada (Base 7)', linewidth=2)
        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Fluxo Espectral (W/m²/Hz)')
        plt.title('Emissão Sincrotron Artística - Magnetosfera de Saturno')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig(filename)
        plt.close()
        print(f"[Base 7] Radiative visualization saved to {filename}")

if __name__ == "__main__":
    transmitter = SynchrotronArtisticTransmitter()
    transmitter.visualize_transmission()
