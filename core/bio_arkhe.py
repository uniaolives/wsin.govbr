"""
BIO-ARKHE v3.0 - DNA e Campo Morfogenético
Sem dependências externas para evitar circular import
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import random

@dataclass
class ArkheGenome:
    """DNA quaternário"""
    C: float  # Chemistry (0-1)
    I: float  # Information (0-1)
    E: float  # Energy (0-1)
    F: float  # Function (0-1)

    def to_vector(self) -> np.ndarray:
        return np.array([self.C, self.I, self.E, self.F], dtype=np.float32)

class MorphogeneticField:
    """Campo escalar 3D com difusão otimizada"""

    def __init__(self, size: Tuple[int, int, int] = (100, 100, 100)):
        self.size = size
        self.grid = np.zeros(size, dtype=np.float32)
        # Kernel de difusão 3x3x3
        self.kernel = self._create_kernel()

    def _create_kernel(self) -> np.ndarray:
        k = np.zeros((3, 3, 3), dtype=np.float32)
        k[1, 1, 1] = 0.4  # Centro
        # Faces
        k[0, 1, 1] = k[2, 1, 1] = k[1, 0, 1] = k[1, 2, 1] = k[1, 1, 0] = k[1, 1, 2] = 0.1
        return k

    def diffuse(self):
        """Difusão manual sem scipy (mais lento mas sem dependência)"""
        new_grid = np.zeros_like(self.grid)
        sx, sy, sz = self.size

        # Convolução manual nas bordas internas (vetorizada com slices para performance)
        new_grid[1:-1, 1:-1, 1:-1] = (
            self.grid[1:-1, 1:-1, 1:-1] * 0.4 +
            self.grid[0:-2, 1:-1, 1:-1] * 0.1 + self.grid[2:, 1:-1, 1:-1] * 0.1 +
            self.grid[1:-1, 0:-2, 1:-1] * 0.1 + self.grid[1:-1, 2:, 1:-1] * 0.1 +
            self.grid[1:-1, 1:-1, 0:-2] * 0.1 + self.grid[1:-1, 1:-1, 2:] * 0.1
        )

        self.grid = new_grid * 0.96  # Decaimento

    def add_signal(self, position: np.ndarray, strength: float):
        x, y, z = position.astype(int)
        if 0 <= x < self.size[0] and 0 <= y < self.size[1] and 0 <= z < self.size[2]:
            self.grid[x, y, z] += strength

    def get_signal_at(self, position: np.ndarray) -> float:
        x, y, z = position.astype(int)
        if 0 <= x < self.size[0] and 0 <= y < self.size[1] and 0 <= z < self.size[2]:
            return float(self.grid[x, y, z])
        return 0.0

    def get_gradient(self, position: np.ndarray) -> np.ndarray:
        x, y, z = position.astype(int)
        x = max(1, min(self.size[0]-2, x))
        y = max(1, min(self.size[1]-2, y))
        z = max(1, min(self.size[2]-2, z))

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
        self.genome = genome

        # Estado fisiológico
        self.health = 0.5 + genome.E * 0.5
        self.age = 0.0
        self.alive = True

        # Conectividade social
        self.neighbors: List[int] = []
        self.bond_strengths: Dict[int, float] = {}

        # Cérebro (inicializado externamente)
        self.brain = None
        self.state = "exploring"
        self.last_decision = ""

    def attach_brain(self, brain):
        self.brain = brain

    def update_physics(self, dt: float, field: MorphogeneticField):
        """Atualização física com metabolismo"""
        # Atualiza posição
        self.position += self.velocity * dt

        # Limites do campo (reflexão suave)
        for i in range(3):
            if self.position[i] < 0:
                self.position[i] = 0
                self.velocity[i] *= -0.5
            elif self.position[i] >= field.size[i]:
                self.position[i] = field.size[i] - 1
                self.velocity[i] *= -0.5

        # Fricção
        self.velocity *= 0.92

        # Metabolismo
        speed = np.linalg.norm(self.velocity)
        movement_cost = speed * speed * 0.001
        base_cost = 0.0003 * (1.1 - self.genome.E)
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
        if other_id in self.neighbors:
            self.neighbors.remove(other_id)
            self.bond_strengths.pop(other_id, None)
