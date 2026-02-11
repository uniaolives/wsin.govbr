"""
CONSTRAINT DISCOVERY ENGINE v2.0
Micro-cérebro Hebbiano com memória de trabalho e atenção seletiva
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import deque
import random

@dataclass
class SynapticTrace:
    """Traço de memória com decaimento temporal"""
    partner_signature: str  # Hash do genoma parceiro
    energy_delta: float
    timestamp: float
    context: np.ndarray  # Estado do campo local

    def decay_factor(self, current_time: float, tau: float = 100.0) -> float:
        """Peso da memória decai exponencialmente"""
        return np.exp(-(current_time - self.timestamp) / tau)

class ConstraintLearner:
    """
    Cérebro Hebbiano com:
    - Plasticidade dependente do tempo (STDP simplificado)
    - Atenção seletiva baseada em novelty
    - Meta-aprendizado (aprende a aprender)
    """

    def __init__(self, agent_id: int, genome_vector: np.ndarray = None):
        self.agent_id = agent_id
        self.weights = np.zeros(4, dtype=np.float32)
        self.bias = 0.0

        # Taxa de aprendizado meta-plástica
        self.lr_base = 0.15
        self.lr_current = self.lr_base
        self.lr_adaptation_rate = 0.01

        # Memória de curto prazo (últimas 20 interações)
        self.working_memory: deque[SynapticTrace] = deque(maxlen=20)

        # Memória de longo prazo (consolidação periódica)
        self.consolidated_patterns: Dict[str, float] = {}

        # Estado interno
        self.novelty_threshold = 0.3
        self.last_prediction_error = 0.0
        self.exploration_bonus = 0.5

        # Estatísticas para visualização
        self.metrics = {
            'predictions': 0,
            'surprises': 0,  # Erros de predição grandes
            'successful_bonds': 0,
            'toxic_bonds': 0,
            'total_energy_gained': 0.0,
            'total_energy_lost': 0.0
        }

        # Inicialização baseada no próprio genoma (autoconhecimento)
        if genome_vector is not None:
            self.weights = genome_vector * 0.5  # Preferência inicial por similaridade

    def evaluate_partner(self, partner_genome: 'ArkheGenome',
                        partner_id: int,
                        current_time: float,
                        local_field_strength: float = 0.0) -> Tuple[float, str]:
        """
        Avaliação preditiva com atenção seletiva e contexto ambiental
        """
        features = np.array([partner_genome.C, partner_genome.I,
                           partner_genome.E, partner_genome.F], dtype=np.float32)

        # 1. Predição baseada em pesos (conhecimento generalizado)
        raw_score = np.dot(self.weights, features) + self.bias
        predicted_outcome = np.tanh(raw_score)

        # 2. Recuperação de memória específica (reconhecimento)
        memory_score = self._retrieve_memory(features, current_time)

        # 3. Cálculo de novidade (dopamina virtual)
        novelty = self._compute_novelty(features)

        # 4. Integração bayesiana aproximada
        if memory_score is not None:
            # Memória específica tem peso maior se for recente/confiável
            final_score = 0.6 * memory_score + 0.4 * predicted_outcome
            reasoning = f"Memória({memory_score:+.2f})×0.6 + Pred({predicted_outcome:+.2f})×0.4"
        else:
            final_score = predicted_outcome
            reasoning = f"Predição({predicted_outcome:+.2f}) | Novidade:{novelty:.2f}"

        # 5. Modulação por estado interno e ambiente
        # Agentes famintos são menos exigentes
        urgency_factor = max(0.0, 1.0 - local_field_strength)

        # 6. Exploração controlada (epsilon-greedy adaptativo)
        exploration = (random.random() - 0.5) * 2 * self.exploration_bonus * novelty
        final_score += exploration

        # Atualiza métricas de predição
        self.metrics['predictions'] += 1

        return float(np.clip(final_score, -1.0, 1.0)), reasoning

    def _retrieve_memory(self, features: np.ndarray,
                        current_time: float) -> Optional[float]:
        """Recupera memória relevante com peso temporal"""
        if not self.working_memory:
            return None

        # Busca por similaridade de padrão
        best_match = None
        best_similarity = 0.7  # Threshold mínimo

        for trace in self.working_memory:
            # Decodifica hash do genoma (simplificado)
            trace_features = np.fromstring(trace.partner_signature.replace('_', ','),
                                          sep=',', dtype=np.float32)
            if len(trace_features) == 4:
                similarity = 1.0 - np.linalg.norm(features - trace_features) / 2.0

                if similarity > best_similarity:
                    decay = trace.decay_factor(current_time)
                    if decay > 0.1:  # Memória ainda relevante
                        best_match = trace
                        best_similarity = similarity

        if best_match:
            return float(np.clip(best_match.energy_delta * 5, -1.0, 1.0))
        return None

    def _compute_novelty(self, features: np.ndarray) -> float:
        """Quão novo é este padrão? (incentiva exploração)"""
        if not self.working_memory:
            return 1.0  # Tudo é novo

        # Distância média às memórias existentes
        distances = []
        for trace in self.working_memory:
            trace_features = np.fromstring(trace.partner_signature.replace('_', ','),
                                          sep=',', dtype=np.float32)
            if len(trace_features) == 4:
                distances.append(np.linalg.norm(features - trace_features))

        return float(min(np.mean(distances) / 2.0, 1.0)) if distances else 1.0

    def learn_from_interaction(self, partner_genome: 'ArkheGenome',
                              partner_id: int,
                              energy_delta: float,
                              current_time: float,
                              local_context: np.ndarray = None):
        """
        Aprendizado Hebbiano com recompensa de predição (TD-learning simplificado)
        """
        features = np.array([partner_genome.C, partner_genome.I,
                           partner_genome.E, partner_genome.F], dtype=np.float32)

        # Predição prévia (para calcular erro)
        prev_prediction = np.tanh(np.dot(self.weights, features) + self.bias)

        # Resultado observado normalizado
        observed_outcome = np.clip(energy_delta * 5, -1.0, 1.0)

        # Erro de predição (surpresa)
        prediction_error = observed_outcome - prev_prediction
        self.last_prediction_error = abs(prediction_error)

        # Se erro for grande, é surpresa → maior aprendizado
        surprise_factor = min(abs(prediction_error) * 2, 1.0)

        if surprise_factor > 0.5:
            self.metrics['surprises'] += 1

        # Atualização dos pesos (Regra Delta com momentum)
        delta_w = prediction_error * self.lr_current * surprise_factor * features
        self.weights += delta_w
        self.bias += float(prediction_error * self.lr_current * surprise_factor * 0.5)

        # Regularização suave para evitar overfitting
        self.weights *= 0.999
        self.bias *= 0.999

        # Limites para estabilidade
        self.weights = np.clip(self.weights, -3.0, 3.0)
        self.bias = np.clip(self.bias, -2.0, 2.0)

        # Armazena na memória de trabalho
        genome_hash = f"{partner_genome.C:.3f}_{partner_genome.I:.3f}_{partner_genome.E:.3f}_{partner_genome.F:.3f}"
        trace = SynapticTrace(
            partner_signature=genome_hash,
            energy_delta=energy_delta,
            timestamp=current_time,
            context=local_context if local_context is not None else np.zeros(3)
        )
        self.working_memory.append(trace)

        # Atualiza métricas
        if energy_delta > 0:
            self.metrics['successful_bonds'] += 1
            self.metrics['total_energy_gained'] += energy_delta
            self.exploration_bonus *= 0.99  # Sucesso reduz exploração (exploitation)
        else:
            self.metrics['toxic_bonds'] += 1
            self.metrics['total_energy_lost'] += abs(energy_delta)
            self.exploration_bonus = min(self.exploration_bonus * 1.05, 0.8)  # Fracasso aumenta exploração

        # Meta-aprendizado: ajusta taxa de aprendizado baseado na consistência
        if self.metrics['predictions'] > 10:
            surprise_rate = self.metrics['surprises'] / self.metrics['predictions']
            if surprise_rate > 0.3:  # Mundo muito volátil
                self.lr_current = min(self.lr_current * 1.01, 0.5)
            else:  # Mundo previsível
                self.lr_current = max(self.lr_current * 0.99, 0.05)

    def get_cognitive_state(self) -> dict:
        """Retorna estado cognitivo rico para visualização"""
        total_bonds = self.metrics['successful_bonds'] + self.metrics['toxic_bonds']
        success_rate = self.metrics['successful_bonds'] / max(1, total_bonds)

        # Determina "personalidade" baseada nos pesos dominantes
        traits = []
        labels = ["Química", "Informação", "Energia", "Função"]
        for i, w in enumerate(self.weights):
            if abs(w) > 0.5:
                direction = "atraído por" if w > 0 else "repelido por"
                traits.append(f"{direction} {labels[i]}")

        if not traits:
            traits.append("Explorador equilibrado")

        return {
            'traits': traits,
            'success_rate': float(success_rate),
            'learning_rate': float(self.lr_current),
            'exploration': float(self.exploration_bonus),
            'memory_size': len(self.working_memory),
            'prediction_accuracy': float(1.0 - (self.metrics['surprises'] / max(1, self.metrics['predictions']))),
            'energy_balance': float(self.metrics['total_energy_gained'] - self.metrics['total_energy_lost'])
        }
