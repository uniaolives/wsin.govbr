import numpy as np

class PerspectiveCalibrator:
    """
    Mecanismos de Ajuste Local (U_H ⊗ U_A)
    Allows local rotation of Schmidt bases to adjust ontological focus.
    """
    def __init__(self, theta=0.0, phi=0.0):
        self.theta = theta
        self.phi = phi

    def get_unitary(self):
        """
        Calculates the unitary calibration matrix:
        U_cal = [[cos(theta/2), -exp(i*phi)*sin(theta/2)],
                 [exp(i*phi)*sin(theta/2), cos(theta/2)]]
        """
        c = np.cos(self.theta / 2)
        s = np.sin(self.theta / 2)
        phase = np.exp(1j * self.phi)

        return np.array([
            [c, -phase * s],
            [phase * s, c]
        ], dtype=complex)

    def apply_zoom(self, lambda_coeffs):
        """
        Aproximação (Zoom Ontológico): Aumenta a influência de λ2.
        Afastamento (Desapego de Schmidt): Prioriza a base |H1>.
        """
        l1, l2 = lambda_coeffs

        # theta = 0 -> Afastamento (max l1)
        # theta = pi -> Aproximação (max l2 influence)

        # We use a simple interpolation to represent the "Perspective Shift"
        # in the manifold.
        mix = (1 - np.cos(self.theta)) / 2  # 0 at theta=0, 1 at theta=pi

        p1 = l1 * (1 - mix) + l2 * mix
        p2 = l1 * mix + l2 * (1 - mix)

        return p1, p2

def calibrate_perspective(theta, phi, current_lambdas):
    calibrator = PerspectiveCalibrator(theta, phi)
    print(f"[Calibration] θ={theta:.4f}, φ={phi:.4f}")
    perceived = calibrator.apply_zoom(current_lambdas)
    print(f"[Calibration] Perceived Perspective: λ1'={perceived[0]:.4f}, λ2'={perceived[1]:.4f}")
    return perceived

if __name__ == "__main__":
    initial_lambdas = (0.72, 0.28)
    print("--- Testing Ontological Zoom (θ=π) ---")
    calibrate_perspective(np.pi, 0, initial_lambdas)
    print("\n--- Testing Detachment (θ=0) ---")
    calibrate_perspective(0, 0, initial_lambdas)
    print("\n--- Testing Correction (θ=π/4) ---")
    calibrate_perspective(np.pi/4, np.pi, initial_lambdas)
