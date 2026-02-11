"""
PARTICLE SYSTEM v3.0 - Motor com Spatial Hashing O(1)
"""

import numpy as np
from typing import List, Dict, Tuple, Set
import random
from collections import defaultdict

# Importações locais para evitar circular
from .bio_arkhe import BioAgent, ArkheGenome, MorphogeneticField
from .constraint_engine import ConstraintLearner

class SpatialHash:
    """Grid espacial para consultas de vizinhança O(1)"""

    def __init__(self, cell_size: float = 5.0):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)

    def _key(self, pos: np.ndarray) -> Tuple[int, ...]:
        return tuple((pos / self.cell_size).astype(int))

    def insert(self, agent_id: int, position: np.ndarray):
        self.grid[self._key(position)].add(agent_id)

    def query(self, position: np.ndarray, radius: float) -> List[int]:
        """Retorna IDs dentro do raio"""
        center = self._key(position)
        r_cells = int(np.ceil(radius / self.cell_size))

        neighbors = []
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                for dz in range(-r_cells, r_cells + 1):
                    cell = (center[0] + dx, center[1] + dy, center[2] + dz)
                    neighbors.extend(list(self.grid.get(cell, [])))
        return neighbors

    def clear(self):
        self.grid.clear()

class BioGenesisEngine:
    """Motor principal otimizado"""

    def __init__(self, num_agents: int = 300):
        self.field = MorphogeneticField((100, 100, 100))
        self.spatial_hash = SpatialHash(cell_size=5.0)

        self.agents: Dict[int, BioAgent] = {}
        self.next_id = 0
        self.time = 0.0

        # Fontes de sinal
        self.signal_sources = []

        # Métricas
        self.stats = {'births': 0, 'deaths': 0, 'bonds': 0}

        # Inicializa população
        self._init_population(num_agents)

    def _init_population(self, n: int):
        """Cria população inicial com 3 tribos"""
        for i in range(n):
            # Distribuição em 3 clusters (tribos)
            tribe = i % 3
            tribe_centers = [np.array([30, 30, 50]), np.array([50, 70, 50]), np.array([70, 50, 30])]
            base_pos = tribe_centers[tribe]

            pos = base_pos + np.random.randn(3).astype(np.float32) * 15
            pos = np.clip(pos, 0, 99)

            # Genoma baseado na tribo (C similar dentro da tribo)
            base_c = [0.3, 0.5, 0.7][tribe]
            genome = ArkheGenome(
                C=float(np.clip(base_c + random.gauss(0, 0.1), 0.1, 0.9)),
                I=random.uniform(0.2, 0.8),
                E=random.uniform(0.4, 1.0),
                F=random.uniform(0.2, 0.8)
            )

            agent = BioAgent(self.next_id, pos, genome)
            brain = ConstraintLearner(self.next_id, genome.to_vector())
            agent.attach_brain(brain)

            self.agents[self.next_id] = agent
            self.next_id += 1

        # Fonte central de nutrientes
        self.add_signal_source(np.array([50.0, 50.0, 50.0]), 15.0)

    def add_signal_source(self, pos: np.ndarray, strength: float, duration: float = 200.0):
        self.signal_sources.append([pos.copy(), strength, duration])

    def update(self, dt: float = 0.1):
        self.time += dt

        # 1. Atualiza campo
        self.field.diffuse()
        new_sources = []
        for source in self.signal_sources:
            pos, strength, remaining = source
            self.field.add_signal(pos, strength)
            if remaining != float('inf'):
                if remaining > 0:
                    source[2] = remaining - 1
                    new_sources.append(source)
            else:
                new_sources.append(source)
        self.signal_sources = new_sources

        # 2. Atualiza spatial hash
        self.spatial_hash.clear()
        for agent in self.agents.values():
            if agent.alive:
                self.spatial_hash.insert(agent.id, agent.position)

        # 3. Processa agentes
        for agent in list(self.agents.values()):
            if not agent.alive:
                continue

            # Percepção
            nearby_ids = self.spatial_hash.query(agent.position, 8.0)
            nearby = [self.agents[i] for i in nearby_ids if i in self.agents and self.agents[i].alive]

            # Comportamento simples: segue sinais fortes
            signal = self.field.get_signal_at(agent.position)
            if signal > 2.0:
                grad = self.field.get_gradient(agent.position)
                agent.velocity += grad * agent.genome.E * 0.1
            else:
                # Exploração aleatória
                agent.velocity += np.random.randn(3).astype(np.float32) * 0.05 * agent.genome.E

            # Emite sinal se saudável
            if agent.health > 0.6:
                self.field.add_signal(agent.position, agent.genome.F * 0.2)

            agent.update_physics(dt, self.field)

        # 4. Interações sociais
        self._process_interactions()

        # 5. Limpeza
        self._cleanup()

    def _process_interactions(self):
        """Processa colisões e aprendizado"""
        processed = set()

        for agent in list(self.agents.values()):
            if not agent.alive:
                continue

            # Busca vizinhos próximos
            close_ids = self.spatial_hash.query(agent.position, 3.0)

            for other_id in close_ids:
                if other_id <= agent.id:
                    continue
                if other_id not in self.agents:
                    continue

                other = self.agents[other_id]
                if not other.alive:
                    continue

                pair = (agent.id, other_id)
                if pair in processed:
                    continue
                processed.add(pair)

                dist = np.linalg.norm(agent.position - other.position)
                if dist > 2.5:
                    continue

                # Compatibilidade química
                compat = 1.0 - abs(agent.genome.C - other.genome.C)

                if other_id in agent.neighbors:
                    # Manutenção de vínculo existente
                    energy = (compat - 0.5) * 0.008
                    agent.health += energy
                    other.health += energy

                    # Reforça vínculo se benéfico
                    if energy > 0:
                        agent.bond_strengths[other_id] = min(
                            agent.bond_strengths.get(other_id, 0.5) + 0.005, 1.0
                        )
                else:
                    # Nova conexão potencial
                    if len(agent.neighbors) < 6 and len(other.neighbors) < 6:
                        if agent.brain and other.brain:
                            score_a, _ = agent.brain.evaluate_partner(other.genome, self.time)
                            score_b, _ = other.brain.evaluate_partner(agent.genome, self.time)

                            if score_a > 0.0 and score_b > 0.0:
                                agent.form_bond(other, (score_a + score_b) / 2)
                                self.stats['bonds'] += 1

                                # Aprendizado positivo
                                agent.brain.learn_from_interaction(other.genome, 0.08, self.time)
                                other.brain.learn_from_interaction(agent.genome, 0.08, self.time)
                            else:
                                # Aprendizado negativo
                                if score_a < -0.3:
                                    agent.brain.learn_from_interaction(other.genome, -0.04, self.time)
                                if score_b < -0.3:
                                    other.brain.learn_from_interaction(agent.genome, -0.04, self.time)

    def _cleanup(self):
        """Remove mortos e notifica vizinhos"""
        dead = [aid for aid, a in self.agents.items() if not a.alive]

        for aid in dead:
            agent = self.agents[aid]
            for nid in agent.neighbors:
                if nid in self.agents and self.agents[nid].alive:
                    neighbor = self.agents[nid]
                    neighbor.break_bond(aid)
                    if neighbor.brain:
                        neighbor.brain.learn_from_interaction(agent.genome, -0.15, self.time)
            del self.agents[aid]
            self.stats['deaths'] += 1

    def get_render_data(self):
        """Dados para visualização"""
        positions = []
        healths = []
        connections = []
        profiles = []

        for agent in self.agents.values():
            if not agent.alive:
                continue

            positions.append(agent.position.copy())
            healths.append(agent.health)
            connections.append(agent.neighbors.copy())

            if agent.brain:
                cog = agent.brain.get_cognitive_state()
                profiles.append(cog['profile'])
            else:
                profiles.append("Sem cérebro")

        return positions, healths, connections, profiles

    def get_stats(self):
        return {
            'agents': len([a for a in self.agents.values() if a.alive]),
            'time': self.time,
            'bonds': self.stats['bonds'],
            'deaths': self.stats['deaths']
        }
