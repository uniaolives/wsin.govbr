"""
BIO-GÊNESE: Active Component Assembly Engine com Aprendizado
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import random
from dataclasses import dataclass
from .bio_arkhe import BioAgent, ArkheGenome, MorphogeneticField

# Configuração do Sistema
INITIAL_POPULATION = 600
FIELD_SIZE = (100, 100, 100)
SPAWN_RADIUS = 40

@dataclass
class BioState:
    time_step: int = 0
    total_energy: float = 0.0
    structure_coherence: float = 0.0
    cognitive_diversity: float = 0.0
    average_learning: float = 0.0
    successful_interactions: int = 0
    failed_interactions: int = 0

class CognitiveParticleEngine:
    def __init__(self, num_agents: int = INITIAL_POPULATION):
        self.field = MorphogeneticField(size=FIELD_SIZE)
        self.agents: Dict[int, BioAgent] = {}
        self.agent_counter = 0
        self.state = BioState()
        self.signals: Dict[Tuple[int, int, int], float] = {}
        self.simulation_step = 0
        self._create_cognitive_soup(num_agents)
        self._add_signal_source(np.array(FIELD_SIZE) // 2, 15.0)

    def _create_cognitive_soup(self, num_agents: int):
        center = np.array(FIELD_SIZE) // 2
        for i in range(num_agents):
            genome = ArkheGenome(
                C=random.uniform(0.1, 0.9), I=random.uniform(0.1, 0.9),
                E=random.uniform(0.3, 1.0), F=random.uniform(0.1, 0.9)
            )
            theta, phi, r = random.random()*2*np.pi, random.random()*np.pi, random.random()*SPAWN_RADIUS
            x = center[0] + r * np.sin(phi) * np.cos(theta)
            y = center[1] + r * np.sin(phi) * np.sin(theta)
            z = center[2] + r * np.cos(phi)
            agent = BioAgent(self.agent_counter, np.array([x, y, z]), genome)
            self.agents[self.agent_counter] = agent
            self.agent_counter += 1

    def _add_signal_source(self, position: np.ndarray, strength: float):
        x, y, z = position.astype(int)
        self.signals[(x, y, z)] = strength

    def update(self, dt: float):
        self.simulation_step += 1
        self.state.time_step = self.simulation_step
        for agent in self.agents.values():
            agent.prev_health = agent.health

        self._update_morphogenetic_field()

        # Agentes
        for agent in list(self.agents.values()):
            if agent.health <= 0: continue
            sensory = agent.perceive_environment(self.field)
            vel = agent.decide_movement(sensory, self.agents)
            agent.velocity = agent.velocity * 0.8 + vel * 0.2

            # Absorção de sinal
            local_sig = self.field.get_signal_at(agent.position)
            if local_sig > 5.0: agent.health = min(1.5, agent.health + 0.005)

            agent.update_physics(dt)

        self._process_cognitive_interactions()
        self._learning_feedback()
        self._purge_dead_agents()
        self._update_metrics()

    def _update_morphogenetic_field(self):
        self.field.signal_grid.fill(0)
        for (x, y, z), strength in self.signals.items():
            if 0 <= x < 100 and 0 <= y < 100 and 0 <= z < 100:
                self.field.signal_grid[x, y, z] += strength
        for agent in self.agents.values():
            if agent.health > 0:
                pos = agent.position.astype(int)
                if 0 <= pos[0] < 100 and 0 <= pos[1] < 100 and 0 <= pos[2] < 100:
                    self.field.signal_grid[pos[0], pos[1], pos[2]] += agent.genome.F * agent.genome.E
        self.field._diffuse_signal()

    def _process_cognitive_interactions(self):
        agent_list = list(self.agents.values())
        COST = 0.0002
        for i, agent in enumerate(agent_list):
            if agent.health <= 0: continue
            for nid in list(agent.neighbors):
                neighbor = self.agents.get(nid)
                if not neighbor or neighbor.health <= 0:
                    agent.neighbors.remove(nid); continue
                agent.health -= COST
                comp = 1.0 - abs(agent.genome.C - neighbor.genome.C)
                if comp > 0.6: agent.health = min(1.5, agent.health + 0.004 * comp)
                elif comp < 0.3: agent.health -= 0.001

            if len(agent.neighbors) >= 6: continue
            # Amostra para novas conexões
            for other in random.sample(agent_list, min(len(agent_list), 10)):
                if other.id == agent.id or other.health <= 0 or len(other.neighbors) >= 6: continue
                if np.linalg.norm(agent.position - other.position) < 4.0:
                    ok_a, _ = agent.evaluate_connection(other)
                    ok_b, _ = other.evaluate_connection(agent)
                    if ok_a and ok_b:
                        if other.id not in agent.neighbors:
                            agent.neighbors.append(other.id)
                            other.neighbors.append(agent.id)
                            self.state.successful_interactions += 1
                    else:
                        self.state.failed_interactions += 1

    def _learning_feedback(self):
        for agent in self.agents.values():
            if agent.health <= 0: continue
            delta = agent.health - agent.prev_health
            if abs(delta) > 0.0001:
                if agent.neighbors:
                    share = delta / len(agent.neighbors)
                    for nid in agent.neighbors:
                        neighbor = self.agents.get(nid)
                        if neighbor:
                            agent.brain.learn_from_interaction(neighbor.genome, nid, share, self.simulation_step)
                elif abs(delta) > 0.001:
                    agent.brain.learn_from_interaction(ArkheGenome(0.5,0.5,0.5,0.5), -1, delta*2, self.simulation_step)

    def _purge_dead_agents(self):
        dead = [aid for aid, a in self.agents.items() if a.health <= 0]
        for aid in dead: del self.agents[aid]

    def _update_metrics(self):
        if not self.agents: return
        self.state.total_energy = sum(a.health for a in self.agents.values()) / len(self.agents)
        self.state.structure_coherence = sum(len(a.neighbors) for a in self.agents.values()) / (len(self.agents)*6 + 1e-6)
        weights = [a.brain.weights for a in self.agents.values()]
        if len(weights) > 1: self.state.cognitive_diversity = float(np.mean(np.std(weights, axis=0)))
        success = [a.brain.get_cognitive_state()['success_rate'] for a in self.agents.values()]
        self.state.average_learning = float(np.mean(success))

    def get_render_data(self):
        p, e, c, cog = [], [], [], []
        for a in self.agents.values():
            if a.health > 0:
                p.append(a.position.copy()); e.append(a.health); c.append(a.neighbors.copy())
                s_rate = a.brain.get_cognitive_state()['success_rate']
                cog.append("smart" if s_rate > 0.6 else "confused" if s_rate < 0.3 else "neutral")
        return p, e, c, cog

    def get_agent_info(self, aid):
        a = self.agents.get(aid)
        if not a: return None
        return {"id": a.id, "health": a.health, "genome": str(a.genome), "profile": a.brain.get_weights_description(), "mood": a.mood}

    def inject_signal(self, pos, strength=10.0):
        self._add_signal_source(pos, strength)
