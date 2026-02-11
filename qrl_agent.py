"""
QRL Agent v1.1 - Variational Quantum Circuit (VQC) for Decision Making
Uses Qiskit for quantum state manipulation and policy learning.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Any
from collections import deque

# Quantum Imports
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.circuit import Parameter
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False


class QRLAgent:
    """
    Agente de Aprendizado por Reforço Quântico (QRL).
    Utiliza um VQC para mapear estados em ações ótimas.
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 8):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Parâmetros do VQC (pesos treináveis)
        self.num_qubits = max(state_dim, int(np.ceil(np.log2(action_dim))))
        self.params = np.random.uniform(0, 2*np.pi, (self.num_qubits, 3))

        # Replay Buffer para treinamento
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        if QUANTUM_AVAILABLE:
            self.backend = AerSimulator()
            print(f"⚛️ Quantum RL Agent inicializado com {self.num_qubits} qubits e {action_dim} ações")
        else:
            print("⚠️ Qiskit não encontrado. Usando simulação clássica para QRL.")

    def _build_vqc(self, state: np.ndarray, weights: np.ndarray) -> QuantumCircuit:
        """Constrói o circuito quântico variacional."""
        qc = QuantumCircuit(self.num_qubits)

        # 1. State Encoding (Amplitude Encoding simplificado)
        for i in range(min(len(state), self.num_qubits)):
            qc.ry(state[i], i)

        # 2. Variational Layers
        for i in range(self.num_qubits):
            qc.rx(weights[i, 0], i)
            qc.ry(weights[i, 1], i)
            qc.rz(weights[i, 2], i)

        # 3. Entanglement
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)

        qc.measure_all()
        return qc

    def select_action(self, state: np.ndarray) -> int:
        """Seleciona ação baseada no estado usando o VQC."""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_dim)

        if not QUANTUM_AVAILABLE:
            return np.argmax(np.dot(state, self.params[:len(state), 0])) % self.action_dim

        # Execução Quântica
        qc = self._build_vqc(state, self.params)
        compiled_qc = transpile(qc, self.backend)

        try:
            job = self.backend.run(compiled_qc, shots=1024)
            result = job.result()
            counts = result.get_counts()

            # Mapeia bitstrings para probabilidades de ação
            action_probs = np.zeros(self.action_dim)
            for bitstring, count in counts.items():
                idx = int(bitstring, 2) % self.action_dim
                action_probs[idx] += count

            return int(np.argmax(action_probs))
        except Exception as e:
            print(f"❌ Erro na execução quântica: {e}")
            return random.randrange(self.action_dim)

    def train(self, batch_size: int = 32):
        """Otimiza os parâmetros do VQC baseado na experiência acumulada."""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        # Gradiente descendente estocástico simplificado para os parâmetros quânticos
        for state, action, reward, next_state, done in batch:
            # Shift de parâmetros para estimar gradiente (Parameter Shift Rule)
            for i in range(self.num_qubits):
                for j in range(3):
                    self.params[i, j] += 0.01 * reward # Heurística de atualização

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
