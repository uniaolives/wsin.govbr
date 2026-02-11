"""
CONSTRAINT DISCOVERY ENGINE
Implementa aprendizado Hebbiano para avaliação de viabilidade de conexões.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

@dataclass
class ArkheGenome:
    """O DNA do Agente: Define sua personalidade e função."""
    C: float  # Chemistry: Força de ligação (0.0 - 1.0)
    I: float  # Information: Capacidade de processamento
    E: float  # Energy: Mobilidade e influência
    F: float  # Function: Frequência de sinalização

@dataclass
class SynapticMemory:
    """Memória de uma interação específica"""
    partner_id: int
    partner_genome_hash: str
    energy_delta: float
    timestamp: int

class ConstraintLearner:
    """
    O Micro-Cérebro do Agente.
    Aprende a mapear características dos vizinhos -> sobrevivência.

    Princípio: "O que não me mata me fortalece, mas o que me fortalece eu memorizo"
    """

    def __init__(self, agent_id: int, learning_rate: float = 0.1):
        self.agent_id = agent_id

        # Pesos sinápticos para [C, I, E, F]
        # Inicialmente neutros (0.0) - Tabula Rasa
        self.weights = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.bias = 0.0
        self.base_learning_rate = learning_rate

        # Memória episódica
        self.memories: List[SynapticMemory] = []
        self.max_memories = 50

        # Estatísticas
        self.successful_bonds = 0
        self.toxic_bonds = 0
        self.total_energy_gained = 0.0
        self.total_energy_lost = 0.0

        # Estado emocional (simplificado)
        self.trust_level = 0.5  # 0.0 = paranóico, 1.0 = confiante
        self.curiosity = 0.7    # 0.0 = conservador, 1.0 = explorador

    def evaluate_partner(self, partner_genome: ArkheGenome, partner_id: int = -1) -> Tuple[float, str]:
        """
        Prevê o valor de uma conexão com base no genoma do parceiro.
        """
        C, I, E, F = partner_genome.C, partner_genome.I, partner_genome.E, partner_genome.F
        features = np.array([C, I, E, F], dtype=np.float32)

        # Verifica memória específica deste parceiro
        memory_score = self._check_memory(partner_id, partner_genome)

        # Score baseado em pesos sinápticos
        synaptic_score = np.dot(self.weights, features) + self.bias
        synaptic_score = np.tanh(synaptic_score)

        # Combina memória específica com aprendizado geral
        if memory_score is not None:
            final_score = 0.7 * memory_score + 0.3 * synaptic_score
            reasoning = f"Memória específica: {memory_score:.2f} + Sináptico: {synaptic_score:.2f}"
        else:
            final_score = synaptic_score
            reasoning = f"Baseado em padrões gerais: {synaptic_score:.2f}"

        # Adiciona curiosidade (exploração)
        exploration_bonus = (random.random() - 0.5) * (1.0 - self.trust_level) * self.curiosity
        final_score += exploration_bonus

        final_score = max(-1.0, min(1.0, final_score))
        return float(final_score), reasoning

    def _check_memory(self, partner_id: int, partner_genome: ArkheGenome) -> Optional[float]:
        """Verifica memórias específicas deste parceiro"""
        if partner_id == -1:
            return None

        recent_memories = [m for m in self.memories if m.partner_id == partner_id]
        if not recent_memories:
            return None

        total_weight = 0.0
        weighted_sum = 0.0
        for i, memory in enumerate(recent_memories[-3:]):
            recency = (i + 1) / len(recent_memories[-3:])
            weighted_sum += memory.energy_delta * recency
            total_weight += recency

        if total_weight > 0:
            return weighted_sum / total_weight * 10.0
        return None

    def learn_from_interaction(self, partner_genome: ArkheGenome, partner_id: int, energy_delta: float, timestamp: int):
        """
        Atualiza os pesos com base no resultado real da interação.
        """
        C, I, E, F = partner_genome.C, partner_genome.I, partner_genome.E, partner_genome.F
        features = np.array([C, I, E, F], dtype=np.float32)

        learning_strength = min(abs(energy_delta) * 5.0, 1.0)
        direction = 1.0 if energy_delta > 0 else -1.0
        effective_lr = self.base_learning_rate * learning_strength

        delta = direction * effective_lr * features
        self.weights += delta
        self.bias += direction * effective_lr * 0.5

        self.weights = np.clip(self.weights, -2.0, 2.0)
        self.bias = np.clip(self.bias, -1.0, 1.0)

        genome_hash = f"{C:.2f}_{I:.2f}_{E:.2f}_{F:.2f}"
        memory = SynapticMemory(
            partner_id=partner_id,
            partner_genome_hash=genome_hash,
            energy_delta=energy_delta,
            timestamp=timestamp
        )

        self.memories.append(memory)
        if len(self.memories) > self.max_memories:
            self.memories.pop(0)

        if energy_delta > 0:
            self.successful_bonds += 1
            self.total_energy_gained += energy_delta
            self.trust_level = min(1.0, self.trust_level + 0.01)
        else:
            self.toxic_bonds += 1
            self.total_energy_lost += abs(energy_delta)
            self.trust_level = max(0.0, self.trust_level - 0.02)

        success_ratio = self.successful_bonds / max(1, self.successful_bonds + self.toxic_bonds)
        self.curiosity = 0.3 + success_ratio * 0.5

    def get_cognitive_state(self) -> dict:
        """Retorna o estado cognitivo atual do agente"""
        preferences = []
        labels = ["Química(C)", "Informação(I)", "Energia(E)", "Função(F)"]

        for i, weight in enumerate(self.weights):
            if weight > 0.3:
                preferences.append(f"Gosta de {labels[i]}")
            elif weight < -0.3:
                preferences.append(f"Evita {labels[i]}")

        if not preferences:
            preferences.append("Neutro/Explorando")

        return {
            "preferences": preferences,
            "trust": float(self.trust_level),
            "curiosity": float(self.curiosity),
            "success_rate": self.successful_bonds / max(1, self.successful_bonds + self.toxic_bonds),
            "total_energy_gained": float(self.total_energy_gained),
            "memories_count": len(self.memories)
        }

    def get_weights_description(self) -> str:
        """Descreve o perfil cognitivo baseado nos pesos"""
        max_idx = np.argmax(np.abs(self.weights))
        labels = ["Química(C)", "Informação(I)", "Energia(E)", "Função(F)"]

        if abs(self.weights[max_idx]) < 0.2:
            return "Explorador Inexperiente"

        if self.weights[max_idx] > 0:
            return f"Busca {labels[max_idx]}"
        else:
            return f"Evita {labels[max_idx]}"
