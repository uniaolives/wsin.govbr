"""
CONSTRAINT DISCOVERY ENGINE
Implementa aprendizado Hebbiano para avaliação de viabilidade de conexões.
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class ArkheGenome:
    """O DNA do Agente: Define sua personalidade e função."""
    C: float  # Chemistry: Força de ligação (0.0 - 1.0)
    I: float  # Information: Capacidade de processamento
    E: float  # Energy: Mobilidade e influência
    F: float  # Function: Frequência de sinalização

@dataclass
class InteractionExperience:
    """Registro de uma interação passada"""
    partner_genome: np.ndarray  # Vetor [C, I, E, F] do parceiro
    energy_delta: float         # Resultado energético (+ ou -)

class ConstraintLearner:
    """
    O Micro-Cérebro do Agente.
    Aprende a mapear características dos vizinhos -> sobrevivência.
    """
    def __init__(self, learning_rate=0.1):
        # Pesos para [C, I, E, F]
        # Inicialmente aleatórios (tabula rasa)
        self.weights = np.random.uniform(-0.5, 0.5, 4)
        self.bias = 0.0
        self.learning_rate = learning_rate

        # Estatísticas
        self.successful_bonds = 0
        self.toxic_bonds = 0

    def evaluate_partner(self, partner_genome: ArkheGenome) -> float:
        """
        Prevê o valor de uma conexão com base no genoma do parceiro.
        Retorna: Score de viabilidade (-1.0 a 1.0)
        """
        # Converte genoma para vetor
        features = np.array([
            partner_genome.C,
            partner_genome.I,
            partner_genome.E,
            partner_genome.F
        ])

        # Produto escalar (Perceptron simples)
        # Score = w1*C + w2*I + w3*E + w4*F + bias
        score = np.dot(self.weights, features) + self.bias
        return np.tanh(score)  # Normaliza entre -1 e 1

    def learn(self, partner_genome: ArkheGenome, energy_delta: float):
        """
        Atualiza os pesos com base no resultado real da interação.
        Regra Hebbiana: "Neurons that fire together, wire together" (se o resultado for positivo)
        """
        features = np.array([
            partner_genome.C,
            partner_genome.I,
            partner_genome.E,
            partner_genome.F
        ])

        # O sinal do aprendizado depende se ganhamos ou perdemos energia
        # Se energy_delta > 0: Reforça a conexão com essas características
        # Se energy_delta < 0: Inibe a conexão

        learning_signal = np.clip(energy_delta * 10.0, -1.0, 1.0)

        self.weights += self.learning_rate * learning_signal * features
        self.bias += self.learning_rate * learning_signal

        # Estatísticas
        if energy_delta > 0:
            self.successful_bonds += 1
        else:
            self.toxic_bonds += 1

    def get_brain_state(self) -> str:
        """Retorna uma string descrevendo o que o agente 'gosta'"""
        labels = ['Química', 'Info', 'Energia', 'Função']

        max_idx = np.argmax(self.weights)
        if self.weights[max_idx] > 0.2:
            return f"Busca: {labels[max_idx]}"
        elif self.weights[max_idx] < -0.2:
            return f"Evita: {labels[max_idx]}"
        return "Explorando"
