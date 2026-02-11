"""
BIO-GÊNESE ENGINE v2.0
Sistema de partículas cognitivas com spatial hashing para performance
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Any
import random
from collections import defaultdict
from .bio_arkhe import BioAgent, ArkheGenome, MorphogeneticField

class SpatialHash:
    """Grid espacial para consultas O(1) de vizinhança"""

    def __init__(self, cell_size: float = 5.0, field_size: Tuple[int, ...] = (100, 100, 100)):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)
        self.field_size = field_size

    def _get_cell(self, position: np.ndarray) -> Tuple[int, ...]:
        return tuple((position / self.cell_size).astype(int))

    def insert(self, agent_id: int, position: np.ndarray):
        cell = self._get_cell(position)
        self.grid[cell].add(agent_id)

    def query(self, position: np.ndarray, radius: float) -> List[int]:
        """Retorna IDs de agentes dentro do raio"""
        center_cell = self._get_cell(position)
        radius_cells = int(np.ceil(radius / self.cell_size))

        neighbors = []
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                for dz in range(-radius_cells, radius_cells + 1):
                    cell = (center_cell[0] + dx, center_cell[1] + dy, center_cell[2] + dz)
                    if cell in self.grid:
                        neighbors.extend(list(self.grid[cell]))
        return neighbors

    def clear(self):
        self.grid.clear()

class BioGenesisEngine:
    """Motor principal com otimizações de performance"""

    def __init__(self, num_agents: int = 400, field_size: Tuple[int, ...] = (100, 100, 100)):
        self.field_size = field_size
        self.field = MorphogeneticField(field_size)
        self.spatial_hash = SpatialHash(cell_size=5.0, field_size=field_size)

        self.agents: Dict[int, BioAgent] = {}
        self.agent_counter = 0
        self.time = 0.0

        # Fontes de sinal dinâmicas
        self.signal_sources: List[List[Any]] = [] # [pos, strength, duration]

        # Importações locais para evitar circular
        from .constraint_engine import ConstraintLearner

        # Inicialização da população
        self._initialize_population(num_agents, ConstraintLearner)

        # Métricas do sistema
        self.metrics = {
            'births': 0,
            'deaths': 0,
            'bonds_formed': 0,
            'bonds_broken': 0,
            'total_energy': 0.0
        }

    def _initialize_population(self, num_agents: int, ConstraintLearner):
        """Cria população inicial com distribuição variada"""
        center = np.array(self.field_size) / 2

        for _ in range(num_agents):
            pos = center + np.random.randn(3).astype(np.float32) * 20
            pos = np.clip(pos, 0, np.array(self.field_size) - 1)

            tribe_c = random.choice([0.2, 0.5, 0.8])

            genome = ArkheGenome(
                C=float(np.clip(tribe_c + random.gauss(0, 0.1), 0.1, 0.9)),
                I=random.uniform(0.2, 0.8),
                E=random.uniform(0.3, 1.0),
                F=random.uniform(0.1, 0.9)
            )

            agent = BioAgent(self.agent_counter, pos, genome)
            brain = ConstraintLearner(self.agent_counter, genome.to_vector())
            agent.attach_brain(brain)

            self.agents[self.agent_counter] = agent
            self.agent_counter += 1

        self.add_signal_source(center, 20.0, float('inf'))

    def add_signal_source(self, position: np.ndarray, strength: float, duration: float = 100.0):
        """Adiciona fonte de sinal temporária ou permanente"""
        self.signal_sources.append([position.copy(), strength, duration])

    def update(self, dt: float = 0.1):
        """Loop principal de simulação"""
        self.time += dt

        # 1. Atualiza campo morfogenético
        self._update_field()

        # 2. Atualiza spatial hash
        self.spatial_hash.clear()
        for agent in self.agents.values():
            if agent.alive:
                self.spatial_hash.insert(agent.id, agent.position)

        # 3. Processa agentes
        for agent in list(self.agents.values()):
            if not agent.alive:
                continue

            nearby_ids = self.spatial_hash.query(agent.position, radius=10.0)
            nearby_agents = [self.agents[i] for i in nearby_ids if i in self.agents and self.agents[i].alive]

            perception = agent.perceive(self.field, nearby_agents)
            acceleration = agent.decide(perception, self.time)
            agent.acceleration = acceleration

            agent.update_physics(dt, self.field)

            if agent.health > 0.6 and agent.genome.F > 0.5:
                self.field.add_signal(agent.position, agent.genome.F * 0.3)

        # 4. Interações e aprendizado
        self._process_interactions()

        # 5. Reprodução seletiva
        self._process_reproduction()

        # 6. Limpeza e métricas
        self._cleanup()
        self._update_metrics()

    def _update_field(self):
        """Atualiza sinais do ambiente"""
        self.field.diffuse()
        new_sources = []
        for source in self.signal_sources:
            pos, strength, remaining = source
            self.field.add_signal(pos, strength)
            if remaining != float('inf'):
                remaining -= 1
                if remaining > 0:
                    source[2] = remaining
                    new_sources.append(source)
            else:
                new_sources.append(source)
        self.signal_sources = new_sources

    def _process_interactions(self):
        """Processa colisões e aprendizado (O(n) com spatial hash)"""
        processed_pairs = set()

        for agent in list(self.agents.values()):
            if not agent.alive:
                continue

            close_ids = self.spatial_hash.query(agent.position, radius=3.0)

            for other_id in close_ids:
                if other_id <= agent.id:
                    continue
                if other_id not in self.agents:
                    continue

                other = self.agents[other_id]
                if not other.alive:
                    continue

                pair = (agent.id, other_id)
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)

                dist = np.linalg.norm(agent.position - other.position)

                if dist < 2.0:
                    compatibility = 1.0 - abs(agent.genome.C - other.genome.C)

                    if other_id in agent.neighbors:
                        energy_exchange = (compatibility - 0.5) * 0.01
                        agent.health += energy_exchange
                        other.health += energy_exchange

                        if energy_exchange > 0:
                            agent.bond_strengths[other_id] = min(
                                agent.bond_strengths.get(other_id, 0.5) + 0.01, 1.0
                            )
                    else:
                        if agent.brain and other.brain:
                            score_a, _ = agent.brain.evaluate_partner(
                                other.genome, other.id, self.time
                            )
                            score_b, _ = other.brain.evaluate_partner(
                                agent.genome, agent.id, self.time
                            )

                            if score_a > 0.1 and score_b > 0.1 and len(agent.neighbors) < 6 and len(other.neighbors) < 6:
                                agent.form_bond(other, strength=float(min(score_a, score_b)))
                                self.metrics['bonds_formed'] += 1

                                agent.brain.learn_from_interaction(
                                    other.genome, other.id, 0.1, self.time
                                )
                                other.brain.learn_from_interaction(
                                    agent.genome, agent.id, 0.1, self.time
                                )
                            else:
                                if score_a < -0.2:
                                    agent.brain.learn_from_interaction(
                                        other.genome, other.id, -0.05, self.time
                                    )
                                if score_b < -0.2:
                                    other.brain.learn_from_interaction(
                                        agent.genome, agent.id, -0.05, self.time
                                    )

    def _process_reproduction(self):
        """Reprodução assexuada de agentes muito saudáveis"""
        new_agents = []

        for agent in list(self.agents.values()):
            if agent.health > 1.5 and agent.age > 50:
                agent.health *= 0.6
                child_genome = agent.genome.mutate(rate=0.05)
                child_pos = agent.position + np.random.randn(3).astype(np.float32) * 2

                child = BioAgent(self.agent_counter, child_pos, child_genome)
                from .constraint_engine import ConstraintLearner
                brain = ConstraintLearner(self.agent_counter, child_genome.to_vector())
                child.attach_brain(brain)

                new_agents.append((self.agent_counter, child))
                self.agent_counter += 1
                self.metrics['births'] += 1

        for aid, agent in new_agents:
            self.agents[aid] = agent

    def _cleanup(self):
        """Remove agentes mortos e conexões inválidas"""
        dead = [aid for aid, a in self.agents.items() if not a.alive]

        for aid in dead:
            agent = self.agents[aid]
            for nid in agent.neighbors:
                if nid in self.agents and self.agents[nid].alive:
                    neighbor = self.agents[nid]
                    neighbor.break_bond(aid)
                    if neighbor.brain:
                        neighbor.brain.learn_from_interaction(
                            agent.genome, aid, -0.2, self.time
                        )
            del self.agents[aid]
            self.metrics['deaths'] += 1

    def _update_metrics(self):
        """Atualiza estatísticas globais"""
        if self.agents:
            self.metrics['total_energy'] = float(sum(a.health for a in self.agents.values()) / len(self.agents))

    def get_render_data(self) -> Tuple[List, List, List, List, List]:
        """Dados para visualização"""
        positions, energies, connections, cognitive_states, colors = [], [], [], [], []

        for agent in self.agents.values():
            if not agent.alive:
                continue

            positions.append(agent.position.copy().tolist())
            energies.append(float(agent.health))
            connections.append([(agent.id, nid) for nid in agent.neighbors])

            if agent.brain:
                cog = agent.brain.get_cognitive_state()
                cognitive_states.append(cog)
                success = cog['success_rate']
                exploration = cog['exploration']
                r = float(1.0 - success)
                g = float(success)
                b = float(exploration)
                colors.append((r, g, b))
            else:
                cognitive_states.append({})
                colors.append((0.5, 0.5, 0.5))

        return positions, energies, connections, cognitive_states, colors

    def get_system_state(self) -> dict:
        """Estado geral do sistema para UI"""
        return {
            'time': float(self.time),
            'population': len([a for a in self.agents.values() if a.alive]),
            'avg_energy': float(self.metrics['total_energy']),
            'births': self.metrics['births'],
            'deaths': self.metrics['deaths'],
            'bonds': self.metrics['bonds_formed']
        }
