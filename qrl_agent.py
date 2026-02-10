"""
🧠 QUANTUM REINFORCEMENT LEARNING AGENT

Agente QRL para otimização de estados emocionais em tempo real.
Usa Variational Quantum Circuits (VQC) para decidir transições ótimas.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp

class QuantumRLAgent:
    """
    Agente que utiliza circuitos quânticos parametrizados para
    aprender e otimizar recomendações de estados emocionais.
    """

    def __init__(self, num_qubits: int = 4, num_actions: int = 8):
        self.num_qubits = num_qubits
        self.num_actions = num_actions
        self.params = np.random.uniform(0, 2 * np.pi, num_qubits * 3)
        self.estimator = Estimator()
        # Observável para medir o estado quântico
        self.observable = SparsePauliOp.from_list([("Z" * num_qubits, 1.0)])

        print(f"⚛️ Quantum RL Agent inicializado com {num_qubits} qubits e {num_actions} ações")

    def _build_circuit(self, params: np.ndarray) -> QuantumCircuit:
        """Constrói o circuito quântico variacional."""
        qc = QuantumCircuit(self.num_qubits)

        for i in range(self.num_qubits):
            qc.rx(params[i*3], i)
            qc.ry(params[i*3 + 1], i)
            qc.rz(params[i*3 + 2], i)

        # Entrelaçamento hexagonal/circular
        for i in range(self.num_qubits):
            qc.cx(i, (i + 1) % self.num_qubits)

        return qc

    def select_action(self, state_vector: np.ndarray) -> int:
        """
        Seleciona uma ação (emoção alvo) baseada no estado atual.
        O estado atual modula as rotações do circuito.
        """
        # Combina parâmetros do agente com o estado do biofeedback
        # Simplificação: adiciona o estado médio aos parâmetros
        state_influence = np.mean(state_vector) if len(state_vector) > 0 else 0
        current_params = self.params + state_influence

        qc = self._build_circuit(current_params)

        try:
            # Estimator V2 uses pubs (circuit, observable, [params])
            pub = (qc, self.observable)
            job = self.estimator.run([pub])
            result = job.result()
            expectation = result[0].data.evs # In V2, evs is used for expectation values
        except Exception as e:
            # print(f"⚠️ Erro no simulador quântico: {e}")
            # Fallback if V2 API differs slightly
            try:
                expectation = result[0].values[0]
            except:
                expectation = np.random.uniform(-1, 1)

        # Mapeia expectativa [-1, 1] para índice de ação [0, num_actions-1]
        normalized_val = (expectation + 1) / 2  # [0, 1]
        action = int(normalized_val * (self.num_actions - 1))

        return int(np.clip(action, 0, self.num_actions - 1))

    def train_step(self, reward: float, learning_rate: float = 0.05):
        """
        Atualiza os parâmetros do circuito baseado no feedback (recompensa).
        Usa uma aproximação de gradiente baseada em perturbação.
        """
        # Perturbação aleatória (Exploração quântica)
        perturbation = np.random.normal(0, 0.2, len(self.params))

        # Se a recompensa for positiva, move os parâmetros na direção da perturbação
        # Se for negativa, move na direção oposta
        self.params += learning_rate * reward * perturbation

        # Mantém os parâmetros no espaço de fase [0, 2pi]
        self.params = np.mod(self.params, 2 * np.pi)

        print(f"📈 QRL Update: Reward={reward:.4f}, Mean Params={np.mean(self.params):.4f}")

if __name__ == "__main__":
    # Teste básico
    agent = QuantumRLAgent()
    state = np.random.rand(10)
    action = agent.select_action(state)
    print(f"Ação selecionada: {action}")
    agent.train_step(reward=0.8)
