"""
BIO-ARKHE: O Protocolo de Vida Digital
Implementação dos 5 Princípios Biológicos de Inteligência
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import random
from .constraint_engine import ConstraintLearner, ArkheGenome

# Constantes Fundamentais
MAX_NEIGHBORS = 6
SIGNAL_DECAY = 0.95
ASSEMBLY_THRESHOLD = 0.8

class MorphogeneticField:
    """Campo Morfogenético - O Meio Ambiente Inteligente"""

    def __init__(self, size=(100, 100, 100)):
        self.size = size
        self.signal_grid = np.zeros(size, dtype=np.float32)
        self.signal_history = []

    def get_local_gradient(self, position: np.ndarray) -> np.ndarray:
        x, y, z = position.astype(int)
        x = max(1, min(self.size[0] - 2, x))
        y = max(1, min(self.size[1] - 2, y))
        z = max(1, min(self.size[2] - 2, z))

        dx = (self.signal_grid[x+1, y, z] - self.signal_grid[x-1, y, z]) / 2.0
        dy = (self.signal_grid[x, y+1, z] - self.signal_grid[x, y-1, z]) / 2.0
        dz = (self.signal_grid[x, y, z+1] - self.signal_grid[x, y, z-1]) / 2.0

        gradient = np.array([dx, dy, dz], dtype=np.float32)
        norm = np.linalg.norm(gradient)
        return gradient / norm if norm > 1e-6 else np.zeros(3, dtype=np.float32)

    def get_signal_at(self, position: np.ndarray) -> float:
        x, y, z = position.astype(int)
        if 0 <= x < self.size[0] and 0 <= y < self.size[1] and 0 <= z < self.size[2]:
            return float(self.signal_grid[x, y, z])
        return 0.0

    def _diffuse_signal(self):
        # Difusão volumétrica simplificada
        self.signal_grid *= SIGNAL_DECAY
        # Em produção, usaria scipy.ndimage.gaussian_filter

class BioAgent:
    """Célula Autônoma com Cérebro Hebbiano Incorporado"""

    def __init__(self, id: int, position: np.ndarray, genome: ArkheGenome, velocity: np.ndarray = None):
        self.id = id
        self.position = position.astype(np.float32)
        self.velocity = velocity if velocity is not None else np.zeros(3, dtype=np.float32)
        self.genome = genome

        # Estado físico
        self.neighbors: List[int] = []
        self.health = 1.0
        self.age = 0
        self.prev_health = 1.0

        # Cérebro Hebbiano
        lr = 0.05 + (self.genome.I * 0.2)
        self.brain = ConstraintLearner(agent_id=id, learning_rate=lr)

        # Comportamento
        self.mood = "curious"
        self.last_action = "none"
        self.decision_reasoning = ""

    def perceive_environment(self, field: MorphogeneticField) -> Dict[str, Any]:
        signal = field.get_signal_at(self.position)
        gradient = field.get_local_gradient(self.position)
        return {'signal_strength': signal, 'signal_gradient': gradient}

    def decide_movement(self, sensory_data: Dict[str, Any], other_agents: Dict[int, 'BioAgent']) -> np.ndarray:
        gradient = sensory_data['signal_gradient']

        if self.mood == "social" or self.genome.C > 0.7:
            social_vector = self._calculate_social_vector(other_agents)
            if np.linalg.norm(social_vector) > 0.1:
                combined = gradient * 0.3 + social_vector * 0.7
                norm = np.linalg.norm(combined)
                if norm > 1e-6: combined /= norm
                self.last_action = "seeking_social"
                return combined * self.genome.E

        if np.linalg.norm(gradient) > 0.1:
            self.last_action = "following_gradient"
            return gradient * self.genome.E
        else:
            random_dir = np.random.randn(3).astype(np.float32)
            norm = np.linalg.norm(random_dir)
            if norm > 1e-6: random_dir /= norm
            self.last_action = "exploring"
            return random_dir * self.genome.E * 0.5

    def _calculate_social_vector(self, other_agents: Dict[int, 'BioAgent']) -> np.ndarray:
        social_vector = np.zeros(3, dtype=np.float32)
        count = 0
        # Amostra limitada para performance
        potential_ids = list(other_agents.keys())
        sample_ids = random.sample(potential_ids, min(len(potential_ids), 20))

        for oid in sample_ids:
            if oid == self.id: continue
            other = other_agents[oid]
            diff = other.position - self.position
            dist = np.linalg.norm(diff)
            if 0 < dist < 20.0:
                score, _ = self.brain.evaluate_partner(other.genome, oid)
                if score > 0.1:
                    social_vector += (diff / dist) * min(score, 1.0)
                    count += 1
        return social_vector / count if count > 0 else social_vector

    def evaluate_connection(self, partner: 'BioAgent') -> Tuple[bool, str]:
        score, reasoning = self.brain.evaluate_partner(partner.genome, partner.id)
        threshold = 0.0
        if self.health < 0.3: threshold = -0.3
        elif self.brain.successful_bonds > 5: threshold = 0.2
        return score > threshold, reasoning

    def update_physics(self, dt: float):
        speed = np.linalg.norm(self.velocity)
        max_speed = self.genome.E * 3.0
        if speed > max_speed: self.velocity = self.velocity / speed * max_speed

        self.position += self.velocity * dt
        self.position = np.clip(self.position, 0, 99)
        self.age += dt
        self.health -= 0.0005 * (1.0 - self.genome.E)

        if self.health > 0.8: self.mood = "social" if self.genome.C > 0.5 else "curious"
        elif self.health < 0.3: self.mood = "avoidant"
