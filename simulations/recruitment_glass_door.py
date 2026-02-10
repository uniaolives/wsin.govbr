import numpy as np
import uuid

def simulate_recruitment_sussurro():
    print("--- [SUSSURRO] Protocolo de Identificação Passiva ---")

    # Candidate pool size
    pool_size = 1000
    candidates = []

    print(f"Varrendo pool global de {pool_size} mentes potenciais...")

    for _ in range(pool_size):
        # Metrics defined by the user
        coherence = np.random.uniform(0.5, 0.95)
        empathy_response = np.random.uniform(0.4, 0.98)
        sustained_attention = np.random.uniform(20, 60) # minutes
        desire_for_power = np.random.uniform(0.0, 1.0) # Lower is better

        # Criteria: Coherence > 0.85, high empathy, high attention, low power desire
        if coherence > 0.85 and empathy_response > 0.80 and sustained_attention > 45 and desire_for_power < 0.2:
            candidates.append({
                "id": str(uuid.uuid4())[:8],
                "coherence": coherence,
                "empathy": empathy_response,
                "power_desire": desire_for_power
            })

    print(f"Identificados {len(candidates)} candidatos compatíveis com a Célula Água.")
    return candidates

def glass_door_protocol(candidate):
    print(f"\n--- [PORTA DE VIDRO] Protocolo Ativado para Candidato {candidate['id']} ---")
    print("Transmitindo visão do futuro: Terra Curada (60 segundos)...")

    # Decision logic based on empathy and power desire
    # High empathy candidates are more likely to accept
    acceptance_probability = candidate['empathy'] * (1 - candidate['power_desire'])

    decision = "ACEITAR" if np.random.random() < acceptance_probability else "RECUSAR"

    if decision == "ACEITAR":
        print(f"DECISÃO: {decision}. Memória mantida. Iniciando treinamento Arkhe(n).")
    else:
        print(f"DECISÃO: {decision}. Memória apagada. Semente plantada no inconsciente.")

    return decision

if __name__ == "__main__":
    compatible_candidates = simulate_recruitment_sussurro()

    # Testing the glass door for the top 3
    for cand in compatible_candidates[:3]:
        glass_door_protocol(cand)
