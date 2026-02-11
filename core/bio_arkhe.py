"""
BIO-ARKHE: Active Component Assembly Architecture
Implementação dos 5 Princípios Biológicos de Inteligência.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any

# Constantes de Vida
MAX_NEIGHBORS = 6  # Simetria Hexagonal (Packing eficiente)
SIGNAL_DECAY = 0.95 # O sinal enfraquece com a distância
ASSEMBLY_THRESHOLD = 0.8 # Afinidade necessária para ligação

@dataclass
class ArkheGenome:
    """O DNA do Agente: Define sua personalidade e função."""
    C: float  # Chemistry: Força de ligação (0.0 - 1.0)
    I: float  # Information: Capacidade de processamento
    E: float  # Energy: Mobilidade e influência
    F: float  # Function: Frequência de sinalização

class MorphogeneticField:
    """
    O Meio Ambiente Ativo.
    Mantém o mapa de 'cheiros' (sinais) que guia os agentes.
    """
    def __init__(self, size=(100, 100, 100)):
        self.size = size
        # Grid escalar para sinalização
        self.signal_grid = np.zeros(size, dtype=np.float32)

    def update_field(self, agents: List['BioAgent']):
        """
        Atualiza o campo baseado na emissão (F) de todos os agentes.
        Simula difusão e decaimento.
        """
        # Decaimento natural (entropia)
        self.signal_grid *= SIGNAL_DECAY

        # Agentes emitem sinal na sua posição
        for agent in agents:
            if agent.health > 0:
                # Normaliza posição para o tamanho do grid (0 a size-1)
                pos = agent.position.astype(int)
                x, y, z = pos

                # Verifica limites
                if 0 <= x < self.size[0] and 0 <= y < self.size[1] and 0 <= z < self.size[2]:
                    # A força do sinal é baseada na Função (F) e Energia (E)
                    emission = agent.genome.F * agent.genome.E * agent.health
                    self.signal_grid[x, y, z] += emission

    def get_local_gradient(self, position: np.ndarray) -> np.ndarray:
        """Calcula gradiente local do campo de sinal"""
        x, y, z = position.astype(int)

        # Garante que estamos dentro dos limites
        x = max(1, min(self.size[0] - 2, x))
        y = max(1, min(self.size[1] - 2, y))
        z = max(1, min(self.size[2] - 2, z))

        # Calcula gradiente usando diferenças finitas
        dx = (self.signal_grid[x+1, y, z] - self.signal_grid[x-1, y, z]) / 2.0
        dy = (self.signal_grid[x, y+1, z] - self.signal_grid[x, y-1, z]) / 2.0
        dz = (self.signal_grid[x, y, z+1] - self.signal_grid[x, y, z-1]) / 2.0

        gradient = np.array([dx, dy, dz], dtype=np.float32)

        # Normaliza se não for zero
        norm = np.linalg.norm(gradient)
        if norm > 1e-6:
            gradient = gradient / norm

        return gradient

    def get_signal_at(self, position: np.ndarray) -> float:
        """Obtém valor do sinal em posição específica"""
        x, y, z = position.astype(int)
        if 0 <= x < self.size[0] and 0 <= y < self.size[1] and 0 <= z < self.size[2]:
            return self.signal_grid[x, y, z]
        return 0.0

    def _diffuse_signal(self):
        """Aplica difusão simples ao campo de sinal"""
        # Simplificação: cada célula compartilha sinal com vizinhos
        # Em uma implementação real usaríamos algo mais eficiente
        pass # Por enquanto, a difusão está embutida na emissão da vizinhança no motor

class BioAgent:
    """
    A Célula Autônoma.
    """
    def __init__(self, id: int, position: np.ndarray, genome: ArkheGenome, velocity: np.ndarray = None):
        self.id = id
        self.position = position.astype(np.float32)
        self.velocity = velocity if velocity is not None else np.zeros(3, dtype=np.float32)
        self.genome = genome

        # Estado interno
        self.neighbors: List[int] = []
        self.health = 1.0
        self.age = 0

        # Memória de curto prazo
        self.memory: List[Tuple[np.ndarray, float]] = []
        self.memory_capacity = max(3, int(genome.I * 10))

    def sense_environment(self, field: MorphogeneticField) -> Dict[str, Any]:
        """Coleta informações do ambiente"""
        signal = field.get_signal_at(self.position)
        gradient = field.get_local_gradient(self.position)

        # Armazena na memória
        self.memory.append((self.position.copy(), signal))
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)

        return {
            'signal': signal,
            'gradient': gradient,
            'memory': self.memory.copy()
        }

    def decide_action(self, sensory_data: Dict[str, Any], all_agents: Dict[int, 'BioAgent']) -> np.ndarray:
        """Decide ação baseada em percepção e genoma"""
        gradient = sensory_data['gradient']

        # Comportamento baseado no genoma
        if self.genome.C > 0.7:  # Social
            # Busca outros agentes próximos
            avg_pos = np.zeros(3, dtype=np.float32)
            count = 0
            # Amostra para performance
            for other_id in list(all_agents.keys())[:20]:
                if other_id != self.id:
                    other = all_agents[other_id]
                    dist = np.linalg.norm(other.position - self.position)
                    if dist < 20:
                        avg_pos += other.position
                        count += 1

            if count > 0:
                social_vector = (avg_pos / count - self.position)
                social_norm = np.linalg.norm(social_vector)
                if social_norm > 1e-6:
                    social_vector /= social_norm
                gradient = gradient * 0.3 + social_vector * 0.7

        elif self.genome.F > 0.6:  # Explorador
            if np.linalg.norm(gradient) < 0.1:
                random_dir = np.random.randn(3).astype(np.float32)
                gradient = random_dir / (np.linalg.norm(random_dir) + 1e-6)

        # Modifica pela energia
        action = gradient * self.genome.E
        return action

    def update_state(self, action: np.ndarray, dt: float):
        """Atualiza estado físico do agente"""
        self.velocity = self.velocity * 0.85 + action * 0.15

        speed = np.linalg.norm(self.velocity)
        max_speed = self.genome.E * 3.0
        if speed > max_speed:
            self.velocity = self.velocity / speed * max_speed

        self.position += self.velocity * dt * 10.0
        self.age += dt

    def sense_and_act(self, field: MorphogeneticField, all_agents: Dict[int, 'BioAgent']):
        """
        Atalho para o ciclo completo (usado pelo motor simplificado).
        """
        data = self.sense_environment(field)
        action = self.decide_action(data, all_agents)
        self.update_state(action, 0.1) # dt fixo para simplicidade
