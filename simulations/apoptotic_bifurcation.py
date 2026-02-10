import numpy as np
from scipy.integrate import odeint

def apoptotic_dynamics(x, t, alpha, beta, K, n, gamma, pulse=0):
    """
    Bistable model for Apoptotic Activation.
    dx/dt = alpha + (beta * x^n) / (K^n + x^n) - gamma * x + pulse(t)
    """
    # Simple pulse logic: active ligand at t=5
    current_pulse = pulse if 5 < t < 10 else 0
    dxdt = alpha + (beta * x**n) / (K**n + x**n) - gamma * x + current_pulse
    return dxdt

def simulate_bifurcation():
    print("--- Apoptotic Algorithm: Dynamical System (Bifurcation) ---")

    # Parameters
    alpha = 0.05  # Basal activation
    beta = 2.0    # Feedback strength
    K = 0.5       # Half-maximal concentration
    n = 4         # Hill coefficient
    gamma = 1.0   # Degradation rate

    t = np.linspace(0, 50, 500)

    # Scenario 1: No Ligand (Stable Life)
    sol_low = odeint(apoptotic_dynamics, 0.1, t, args=(alpha, beta, K, n, gamma, 0))

    # Scenario 2: Ligand Pulse (Apoptotic Trigger)
    sol_high = odeint(apoptotic_dynamics, 0.1, t, args=(alpha, beta, K, n, gamma, 1.0))

    print(f"Scenario 1 (No Ligand): Final x={sol_low[-1][0]:.4f} (STABLE LIFE)")
    print(f"Scenario 2 (Ligand Pulse): Final x={sol_high[-1][0]:.4f} (APOPTOTIC COLLAPSE)")

    print("\n--- Phase Analysis ---")
    if sol_high[-1][0] > 1.0:
        print(f"[ANALYSIS] Threshold crossed at x > {K}. Current state: {sol_high[-1][0]:.2f}")
        print("[RESULT] Irreversible state change detected. The 'Arkhe(n) Reset' has colapsed the wave function.")
    else:
        print("[ANALYSIS] Feedback loop failed to sustain activation. System returned to basal state.")

if __name__ == "__main__":
    simulate_bifurcation()
