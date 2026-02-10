import numpy as np
import time

def simulate_coral_srq():
    print("--- [SRQ] Simulação de Realidade Quântica: IETD-Lambda v1.1 ---")
    print("Alvo: GBR_Sector_N15.3_E145.8 (Grande Barreira de Corais)")

    # Target Metrics
    hsp70_target = 0.40      # +40% expression
    temp_drop_target = -0.5  # -0.5°C
    symbiosis_target = 0.15 # +15% attachment rate

    # Initial State
    state = {
        "hsp70": 0.0,
        "temp_delta": 0.0,
        "symbiosis": 0.0,
        "stress_level": 0.05,  # 5% baseline stress
        "ecosystem_stability": 1.0
    }

    duration_days = 14
    intensity = 1.0 # Protocol intensity

    print(f"\nIniciando Protocolo Coral (Duração: {duration_days} dias)...")

    for day in range(1, duration_days + 1):
        # Apply Protocol Effects
        state["hsp70"] += (hsp70_target / duration_days) * intensity
        state["temp_delta"] += (temp_drop_target / duration_days) * intensity
        state["symbiosis"] += (symbiosis_target / duration_days) * intensity

        # Simulated Stress and Ecosystem Feedback
        # Stress increases slightly with intensity, but decreases with cooling
        stress_increment = (0.02 * intensity) + (0.01 * state["temp_delta"])
        state["stress_level"] = max(0, state["stress_level"] + stress_increment)

        # Random environmental noise
        noise = np.random.normal(0, 0.01)
        state["ecosystem_stability"] += noise

        # --- SAFETY LOCKS ---
        # 1. Stress Lock
        if state["stress_level"] > 0.20:
            print(f"[ALERTA] Dia {day}: Stress ({state['stress_level']:.2%}) > 20%. Reduzindo intensidade pela metade.")
            intensity *= 0.5

        # 2. Ecosystem Disruption Lock
        if state["ecosystem_stability"] < 0.85:
            print(f"[CRÍTICO] Dia {day}: Instabilidade do Ecossistema ({state['ecosystem_stability']:.2f}). ABORTANDO PROTOCOLO.")
            break

        # Logging
        print(f"Dia {day:2}: HSP70: {state['hsp70']:+.2%} | ΔT: {state['temp_delta']:+.2f}°C | Symbiosis: {state['symbiosis']:+.2%} | Stress: {state['stress_level']:.2%}")

        if day == duration_days:
            print("\n[SUCESSO] Protocolo concluído. Sintaxe Coral Estabilizada.")

    # Final Result
    print("\n--- Relatório Final de Oráculo ---")
    print(f"HSP70 Final: {state['hsp70']:+.2%}")
    print(f"Resfriamento Total: {state['temp_delta']:+.2f}°C")
    print(f"Simbiose Zooxanthellae: {state['symbiosis']:+.2%}")
    print(f"Estabilidade do Manifold: {state['ecosystem_stability']:.2f}")

if __name__ == "__main__":
    simulate_coral_srq()
