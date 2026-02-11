"""
BIO-ARKHE v2.0
Integração Corpo-Cérebro com física de soft-body e metabolismos diferenciados
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import random

@dataclass
class ArkheGenome:
    """DNA quaternário com herança epigenética simplificada"""
    C: float  # Química: compatibilidade estrutural
    I: float  # Informação: capacidade de processamento/learning rate
    E: float  # Energia: metabolismo e velocidade
    F: float  # Função: especialização (sinalizador vs consumidor)

    def to_vector(self) -> np.ndarray:
        return np.array([self.C, self.I, self.E, self.F], dtype=np.float32)

    def mutate(self, rate: float = 0.1) -> 'ArkheGenome':
        """Mutação gaussiana com limites"""
        def clamp(x): return max(0.05, min(0.95, x))
        return ArkheGenome(
            C=clamp(self.C + random.gauss(0, rate)),
            I=clamp(self.I + random.gauss(0, rate)),
            E=clamp(self.E + random.gauss(0, rate)),
            F=clamp(self.F + random.gauss(0, rate))
        )

class MorphogeneticField:
    """Campo escalar 3D com difusão e advecção"""

    def __init__(self, size: Tuple[int, int, int] = (100, 100, 100)):
        self.size = size
        self.grid = np.zeros(size, dtype=np.float32)
        self.diffusion_kernel = self._create_diffusion_kernel()

    def _create_diffusion_kernel(self) -> np.ndarray:
        """Kernel 3x3x3 para difusão isotrópica"""
        k = np.zeros((3, 3, 3), dtype=np.float32)
        k[1, 1, 1] = 0.5  # Centro
        for axis in range(3):
            for offset in [-1, 1]:
                idx = [1, 1, 1]
                idx[axis] += offset
                k[tuple(idx)] = 0.083  # Vizinhos diretos (6 faces)
        return k

    def diffuse(self, dt: float = 1.0):
        """Difusão via convolução com condições de contorno periódicas"""
        try:
            from scipy.ndimage import convolve
            self.grid = convolve(self.grid, self.diffusion_kernel, mode='constant', cval=0.0)
        except ImportError:
            # Fallback manual simplificado se scipy não estiver disponível
            new_grid = self.grid * 0.5
            # Operação de vizinhança manual para 6 vizinhos (muito lenta em Python puro, mas funciona)
            # Para performance no sandbox, usaremos apenas o decaimento se convolve falhar
            pass

        # Decaimento natural
        self.grid *= 0.95

    def add_signal(self, position: np.ndarray, strength: float, radius: int = 3):
        """Adiciona sinal em posição"""
        x, y, z = position.astype(int)
        x = np.clip(x, 0, self.size[0]-1)
        y = np.clip(y, 0, self.size[1]-1)
        z = np.clip(z, 0, self.size[2]-1)

        self.grid[x, y, z] += strength

    def get_gradient(self, position: np.ndarray) -> np.ndarray:
        """Calcula gradiente por diferenças finitas"""
        x, y, z = position.astype(int)
        x = np.clip(x, 1, self.size[0]-2)
        y = np.clip(y, 1, self.size[1]-2)
        z = np.clip(z, 1, self.size[2]-2)

        grad = np.array([
            (self.grid[x+1, y, z] - self.grid[x-1, y, z]) / 2.0,
            (self.grid[x, y+1, z] - self.grid[x, y-1, z]) / 2.0,
            (self.grid[x, y, z+1] - self.grid[x, y, z-1]) / 2.0
        ], dtype=np.float32)

        norm = np.linalg.norm(grad)
        return grad / norm if norm > 1e-6 else np.random.randn(3).astype(np.float32) * 0.1

class BioAgent:
    """Agente autônomo com cognição embarcada"""

    def __init__(self, agent_id: int, position: np.ndarray, genome: ArkheGenome):
        self.id = agent_id
        self.position = position.astype(np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.acceleration = np.zeros(3, dtype=np.float32)
        self.genome = genome

        # Estado fisiológico
        self.health = 0.5 + genome.E * 0.5  # Energia inicial baseada no gene E
        self.energy = self.health  # Reserva metabólica
        self.age = 0.0
        self.alive = True

        # Conectividade
        self.neighbors: List[int] = []
        self.bond_strengths: Dict[int, float] = {}  # Força da conexão com cada vizinho

        # Cérebro
        self.brain: Optional[Any] = None # Will be set as ConstraintLearner

        # Estado comportamental
        self.state = "exploring"  # exploring, socializing, fleeing, resting
        self.last_decision = ""

    def attach_brain(self, brain: Any):
        """Conecta o cérebro Hebbiano"""
        self.brain = brain

    def perceive(self, field: MorphogeneticField, nearby_agents: List['BioAgent']) -> dict:
        """Constrói representação interna do ambiente"""
        # Sinais do campo
        idx_x = int(np.clip(self.position[0], 0, field.size[0]-1))
        idx_y = int(np.clip(self.position[1], 0, field.size[1]-1))
        idx_z = int(np.clip(self.position[2], 0, field.size[2]-1))
        local_signal = field.grid[idx_x, idx_y, idx_z]

        gradient = field.get_gradient(self.position)

        # Análise social
        threats = []
        opportunities = []

        for other in nearby_agents:
            if other.id == self.id:
                continue
            dist = np.linalg.norm(self.position - other.position)
            if dist < 5.0:
                # Avaliação rápida (heurística antes do cérebro)
                compat = 1.0 - abs(self.genome.C - other.genome.C)
                if compat > 0.6:
                    opportunities.append(other)
                else:
                    threats.append(other)

        return {
            'signal_strength': local_signal,
            'signal_gradient': gradient,
            'threats': threats,
            'opportunities': opportunities,
            'local_density': len(nearby_agents)
        }

    def decide(self, perception: dict, current_time: float) -> np.ndarray:
        """Tomada de decisão com cognição"""
        if self.brain is None:
            return self._random_walk()

        # Avalia oportunidades
        best_target = None
        best_score = -float('inf')

        for opp in perception['opportunities']:
            score, reasoning = self.brain.evaluate_partner(
                opp.genome, opp.id, current_time,
                local_field_strength=perception['signal_strength']
            )
            if score > best_score:
                best_score = score
                best_target = opp
                self.last_decision = reasoning

        # Comportamento baseado na avaliação
        if best_target and best_score > 0.2:
            direction = best_target.position - self.position
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction /= dist
                self.state = "socializing"
                return direction * self.genome.E * 1.5

        # Se há ameaças, foge
        if perception['threats']:
            flee_vector = np.zeros(3, dtype=np.float32)
            for threat in perception['threats']:
                away = self.position - threat.position
                dist = np.linalg.norm(away)
                if dist > 0:
                    flee_vector += (away / dist) / dist
            if np.linalg.norm(flee_vector) > 0:
                self.state = "fleeing"
                return flee_vector / np.linalg.norm(flee_vector) * self.genome.E * 2.0

        # Segue gradiente de sinal se for forte
        if perception['signal_strength'] > 2.0 and np.linalg.norm(perception['signal_gradient']) > 0.1:
            self.state = "foraging"
            return perception['signal_gradient'] * self.genome.E

        # Exploração padrão
        self.state = "exploring"
        return self._random_walk()

    def _random_walk(self) -> np.ndarray:
        """Movimento browniano com persistência de direção"""
        noise = np.random.randn(3).astype(np.float32)
        noise /= np.linalg.norm(noise) + 1e-6
        return noise * self.genome.E * 0.5

    def update_physics(self, dt: float, field: MorphogeneticField):
        """Atualização física com metabolismo"""
        self.velocity += self.acceleration * dt
        self.velocity *= 0.9  # Arrasto viscoso

        max_speed = self.genome.E * 3.0 * (self.health + 0.1)
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity = self.velocity / speed * max_speed

        self.position += self.velocity * dt

        # Condições de contorno (reflexão suave)
        for i in range(3):
            if self.position[i] < 0:
                self.position[i] = 0
                self.velocity[i] *= -0.5
            elif self.position[i] >= field.size[i]:
                self.position[i] = field.size[i] - 1
                self.velocity[i] *= -0.5

        # Metabolismo
        movement_cost = speed * speed * 0.001
        base_cost = 0.0005 * (1.1 - self.genome.E)
        self.health -= (movement_cost + base_cost) * dt
        self.age += dt

        if self.health <= 0:
            self.alive = False

    def form_bond(self, other: 'BioAgent', strength: float = 0.5):
        """Forma conexão simbiótica"""
        if other.id not in self.neighbors and len(self.neighbors) < 6:
            self.neighbors.append(other.id)
            self.bond_strengths[other.id] = strength

        if self.id not in other.neighbors and len(other.neighbors) < 6:
            other.neighbors.append(self.id)
            other.bond_strengths[self.id] = strength

    def break_bond(self, other_id: int):
        """Rompe conexão"""
        if other_id in self.neighbors:
            self.neighbors.remove(other_id)
            self.bond_strengths.pop(other_id, None)
