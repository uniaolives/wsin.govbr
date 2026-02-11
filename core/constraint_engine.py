"""
CONSTRAINT ENGINE v3.0 - Micro-cérebro Hebbiano Otimizado
Aprendizado metabólico com memória de trabalho e atenção seletiva
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import deque
import random

@dataclass
class SynapticTrace:
    """Traço de memória com decaimento temporal"""
    partner_signature: str
    energy_delta: float
    timestamp: float

    def decay_factor(self, current_time: float, tau: float = 100.0) -> float:
        return np.exp(-(current_time - self.timestamp) / tau)

class ConstraintLearner:
    """
    Cérebro Hebbiano com:
    - STDP (Spike-Timing Dependent Plasticity) simplificado
    - Atenção seletiva por novidade
    - Meta-aprendizado adaptativo
    """

    def __init__(self, agent_id: int, genome_vector: np.ndarray = None):
        self.agent_id = agent_id

        # Pesos sinápticos para [C, I, E, F]
        self.weights = np.zeros(4, dtype=np.float32)
        self.bias = 0.0

        # Taxa de aprendizado adaptativa
        self.lr_base = 0.15
        self.lr_current = self.lr_base

        # Memória de trabalho (últimas 15 interações)
        self.working_memory: deque[SynapticTrace] = deque(maxlen=15)

        # Estado interno
        self.exploration_rate = 0.3
        self.last_prediction_error = 0.0

        # Estatísticas
        self.metrics = {
            'successful_bonds': 0,
            'toxic_bonds': 0,
            'total_energy_gained': 0.0,
            'total_energy_lost': 0.0,
            'predictions': 0,
            'surprises': 0
        }

        # Autoconhecimento: preferência inicial por similaridade
        if genome_vector is not None:
            self.weights = genome_vector * 0.3

    def evaluate_partner(self, partner_genome, current_time: float) -> Tuple[float, str]:
        """
        Avalia parceiro com integração bayesiana aproximada
        """
        features = np.array([partner_genome.C, partner_genome.I,
                           partner_genome.E, partner_genome.F], dtype=np.float32)

        # 1. Predição baseada em pesos aprendidos
        raw_score = np.dot(self.weights, features) + self.bias
        predicted = np.tanh(raw_score)

        # 2. Recuperação de memória episódica
        memory_score = self._retrieve_memory(features, current_time)

        # 3. Integração (memória específica tem peso maior se existir)
        if memory_score is not None:
            final_score = 0.7 * memory_score + 0.3 * predicted
            reasoning = f"Memória({memory_score:+.2f}) + Intuição({predicted:+.2f})"
        else:
            final_score = predicted
            reasoning = f"Intuição({predicted:+.2f})"

        # 4. Exploração controlada (epsilon-greedy)
        if random.random() < self.exploration_rate:
            noise = (random.random() - 0.5) * 0.4
            final_score += noise
            reasoning += f" [Exploração{noise:+.2f}]"

        self.metrics['predictions'] += 1
        return float(np.clip(final_score, -1.0, 1.0)), reasoning

    def _retrieve_memory(self, features: np.ndarray, current_time: float) -> Optional[float]:
        """Recupera memória relevante com decaimento temporal"""
        if not self.working_memory:
            return None

        best_match = None
        best_score = -float('inf')

        for trace in self.working_memory:
            # Decodifica hash simples
            try:
                parts = trace.partner_signature.split('_')
                if len(parts) == 4:
                    trace_features = np.array([float(p) for p in parts], dtype=np.float32)
                    similarity = 1.0 - np.linalg.norm(features - trace_features) / 2.0

                    if similarity > 0.8:  # Threshold de similaridade
                        decay = trace.decay_factor(current_time)
                        weighted_score = trace.energy_delta * 5 * decay * similarity

                        if weighted_score > best_score:
                            best_score = weighted_score
                            best_match = trace
            except:
                continue

        if best_match:
            return float(np.clip(best_score, -1.0, 1.0))
        return None

    def learn_from_interaction(self, partner_genome, energy_delta: float, current_time: float):
        """
        Aprendizado Hebbiano com recompensa de predição (TD-learning)
        """
        features = np.array([partner_genome.C, partner_genome.I,
                           partner_genome.E, partner_genome.F], dtype=np.float32)

        # Calcula erro de predição (surpresa)
        prev_prediction = np.tanh(np.dot(self.weights, features) + self.bias)
        observed = np.clip(energy_delta * 5, -1.0, 1.0)
        prediction_error = observed - prev_prediction

        self.last_prediction_error = float(abs(prediction_error))
        if abs(prediction_error) > 0.3:
            self.metrics['surprises'] += 1

        # Força do aprendizado proporcional à surpresa
        surprise_factor = min(abs(prediction_error) * 2, 1.0)
        effective_lr = self.lr_current * surprise_factor

        # Atualização dos pesos (Regra Delta)
        if energy_delta > 0:
            self.weights += effective_lr * features
            self.bias += float(effective_lr * 0.5)
            self.metrics['successful_bonds'] += 1
            self.metrics['total_energy_gained'] += energy_delta
            self.exploration_rate *= 0.99  # Sucesso reduz exploração
        else:
            self.weights -= effective_lr * features * 0.5
            self.bias -= float(effective_lr * 0.25)
            self.metrics['toxic_bonds'] += 1
            self.metrics['total_energy_lost'] += abs(energy_delta)
            self.exploration_rate = float(min(self.exploration_rate * 1.02, 0.6))

        # Limites para estabilidade
        self.weights = np.clip(self.weights, -2.5, 2.5)
        self.bias = float(np.clip(self.bias, -1.5, 1.5))

        # Armazena na memória de trabalho
        genome_hash = f"{partner_genome.C:.3f}_{partner_genome.I:.3f}_{partner_genome.E:.3f}_{partner_genome.F:.3f}"
        self.working_memory.append(SynapticTrace(
            partner_signature=genome_hash,
            energy_delta=energy_delta,
            timestamp=current_time
        ))

        # Meta-aprendizado: ajusta LR baseado na volatilidade
        if self.metrics['predictions'] > 20:
            surprise_rate = self.metrics['surprises'] / self.metrics['predictions']
            if surprise_rate > 0.4:
                self.lr_current = float(min(self.lr_current * 1.01, 0.4))
            else:
                self.lr_current = float(max(self.lr_current * 0.99, 0.05))

    def get_cognitive_state(self) -> dict:
        """Retorna estado cognitivo rico"""
        total = self.metrics['successful_bonds'] + self.metrics['toxic_bonds']
        success_rate = self.metrics['successful_bonds'] / max(1, total)

        # Determina traços de personalidade
        traits = []
        labels = ["Química", "Informação", "Energia", "Função"]
        for i, w in enumerate(self.weights):
            if abs(w) > 0.4:
                direction = "atraído por" if w > 0 else "evita"
                traits.append(f"{direction} {labels[i]}")

        if not traits:
            traits.append("Explorador equilibrado")

        # Classificação por experiência
        if total < 5:
            profile = "Neófito"
        elif success_rate > 0.7:
            profile = "Especialista"
        elif success_rate > 0.4:
            profile = "Aprendiz"
        else:
            profile = "Cauteloso"

        return {
            'profile': profile,
            'traits': traits,
            'success_rate': float(success_rate),
            'exploration_rate': float(self.exploration_rate),
            'learning_rate': float(self.lr_current),
            'memory_size': len(self.working_memory),
            'energy_balance': float(self.metrics['total_energy_gained'] - self.metrics['total_energy_lost'])
        }
