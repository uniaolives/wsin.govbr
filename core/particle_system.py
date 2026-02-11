"""
BIO-GÊNESE: Active Component Assembly Engine
Substitui o sistema de partículas estáticas por agentes autônomos.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import random
from dataclasses import dataclass
from .bio_arkhe import BioAgent, ArkheGenome, MorphogeneticField

# Constantes do Sistema Vivo
INITIAL_POPULATION = 800
FIELD_SIZE = (100, 100, 100)
SPAWN_RADIUS = 40
MUTATION_RATE = 0.01

@dataclass
class BioState:
    """Estado global do ecossistema"""
    time_step: int = 0
    total_energy: float = 0.0
    structure_coherence: float = 0.0
    signal_diversity: float = 0.0

class BioParticleEngine:
    """
    Motor principal que orquestra o ecossistema de agentes.
    Implementa os 5 princípios da inteligência biológica.
    """

    def __init__(self, num_agents: int = INITIAL_POPULATION):
        self.field = MorphogeneticField(size=FIELD_SIZE)
        self.agents: Dict[int, BioAgent] = {}
        self.agent_counter = 0
        self.state = BioState()
        self.signals: Dict[Tuple[int, int, int], float] = {}

        # Cria população inicial com diversidade genética
        self._create_primordial_soup(num_agents)

        # Adiciona fonte de sinal central
        self._add_signal_source(np.array(FIELD_SIZE) // 2, 10.0)

    def _create_primordial_soup(self, num_agents: int):
        """Cria a população inicial com diversidade genética"""
        center = np.array(FIELD_SIZE) // 2

        for i in range(num_agents):
            # Posição aleatória em esfera
            theta = random.random() * 2 * np.pi
            phi = random.random() * np.pi
            r = random.random() * SPAWN_RADIUS

            x = center[0] + r * np.sin(phi) * np.cos(theta)
            y = center[1] + r * np.sin(phi) * np.sin(theta)
            z = center[2] + r * np.cos(phi)

            # Genoma diversificado
            genome = ArkheGenome(
                C=random.uniform(0.3, 0.9),  # Química variada
                I=random.uniform(0.1, 0.7),  # Informação
                E=random.uniform(0.4, 1.0),  # Energia
                F=random.uniform(0.1, 0.5),  # Função
            )

            agent = BioAgent(
                id=self.agent_counter,
                position=np.array([x, y, z], dtype=np.float32),
                genome=genome,
                velocity=np.zeros(3, dtype=np.float32)
            )

            self.agents[self.agent_counter] = agent
            self.agent_counter += 1

    def _add_signal_source(self, position: np.ndarray, strength: float):
        """Adiciona fonte de sinal ao campo morfogenético"""
        x, y, z = position.astype(int)
        key = (x, y, z)
        self.signals[key] = strength

    def update(self, dt: float, external_signals: Optional[List] = None):
        """
        Atualiza o ecossistema completo com aprendizado.
        """
        self.state.time_step += 1

        # 0. Guarda saúde anterior para calcular delta depois
        for agent in self.agents.values():
            agent.prev_health = agent.health

        # 1. Processa sinais externos
        if external_signals:
            for signal in external_signals:
                pos = signal.get('position', np.array(FIELD_SIZE) // 2)
                strength = signal.get('strength', 5.0)
                self._add_signal_source(pos, strength)

        # 2. Atualiza campo morfogenético
        self._update_morphogenetic_field()

        # 3. Atualiza cada agente
        self._update_agents(dt)

        # 4. Processa interações INTELIGENTES
        self._process_smart_interactions()

        # 5. Fase de Aprendizado (Feedback Metabólico)
        self._metabolic_feedback()

        # 6. Atualiza métricas do ecossistema
        self._update_ecosystem_metrics()

    def _update_morphogenetic_field(self):
        """Atualiza o campo com sinais de agentes e fontes externas"""
        # Limpa campo anterior
        self.field.signal_grid.fill(0)

        # Adiciona fontes de sinal fixas
        for (x, y, z), strength in self.signals.items():
            if 0 <= x < FIELD_SIZE[0] and 0 <= y < FIELD_SIZE[1] and 0 <= z < FIELD_SIZE[2]:
                self.field.signal_grid[x, y, z] += strength

        # Adiciona emissões dos agentes
        for agent in self.agents.values():
            if agent.health > 0:
                emission = agent.genome.F * agent.genome.E * agent.health
                pos = agent.position.astype(int)
                x, y, z = pos

                # Emite para posição atual
                if 0 <= x < FIELD_SIZE[0] and 0 <= y < FIELD_SIZE[1] and 0 <= z < FIELD_SIZE[2]:
                    self.field.signal_grid[x, y, z] += emission

        # Difusão implícita (opcional, pode ser lenta no CPU)
        # self.field._diffuse_signal()

    def _update_agents(self, dt: float):
        """Atualiza estado de todos os agentes"""
        for agent in list(self.agents.values()):
            if agent.health <= 0:
                continue

            # Ciclo sensorial e de ação
            agent.sense_and_act(self.field, self.agents)

            # Mantém dentro dos limites
            agent.position = np.clip(agent.position, 0, np.array(FIELD_SIZE) - 1)

            # Consome energia
            agent.health -= 0.0005 * (1.0 - agent.genome.E)

            # Recupera energia em áreas de alto sinal
            local_signal = self.field.get_signal_at(agent.position)
            if local_signal > 2.0:
                agent.health += 0.001 * agent.genome.E

            # Limita health
            agent.health = min(1.5, agent.health)

        self._purge_dead_agents()

    def _purge_dead_agents(self):
        dead_ids = [aid for aid, a in self.agents.items() if a.health <= 0.0]
        for aid in dead_ids:
            del self.agents[aid]

    def _process_smart_interactions(self):
        """
        Processa ligações usando o ConstraintLearner (Cérebro).
        """
        agent_list = list(self.agents.values())

        # Custo metabólico de manter uma ligação
        CONNECTION_COST = 0.0005

        for i, agent in enumerate(agent_list):
            if agent.health <= 0: continue

            # A. Processa ligações existentes (Manter ou Cortar?)
            for neighbor_id in list(agent.neighbors):
                neighbor = self.agents.get(neighbor_id)
                if not neighbor or neighbor.health <= 0:
                    if neighbor_id in agent.neighbors:
                        agent.neighbors.remove(neighbor_id)
                    continue

                # Consome energia para manter conexão
                agent.health -= CONNECTION_COST

                # Troca de energia através da conexão (Simbiose vs Parasitismo)
                compatibility = 1.0 - abs(agent.genome.C - neighbor.genome.C)

                if compatibility > 0.5:
                    # Simbiose: boost de energia
                    boost = 0.002 * compatibility
                    agent.health += boost
                else:
                    # Incompatibilidade: Dreno
                    agent.health -= 0.001

            # B. Tenta novas ligações (se tiver espaço)
            if len(agent.neighbors) >= 6: continue

            # Procura candidatos próximos (amostra para performance)
            for j in range(i + 1, min(i + 20, len(agent_list))):
                other = agent_list[j]
                if other.health <= 0 or len(other.neighbors) >= 6: continue

                dist = np.linalg.norm(agent.position - other.position)
                if dist < 4.0:
                    # --- AQUI ENTRA O CÉREBRO ---
                    # Agente avalia o Parceiro
                    prediction = agent.brain.evaluate_partner(other.genome)

                    # Decisão: Só conecta se a previsão for positiva (com algum ruído para exploração)
                    exploration_noise = (random.random() - 0.5) * 0.2

                    if prediction + exploration_noise > 0.0:
                        # Conecta!
                        if other.id not in agent.neighbors:
                            agent.neighbors.append(other.id)
                            other.neighbors.append(agent.id)

    def _metabolic_feedback(self):
        """
        Ensina o cérebro com base na variação de saúde (Delta E).
        """
        for agent in self.agents.values():
            if agent.health <= 0: continue

            # Variação de energia neste frame
            delta_e = agent.health - agent.prev_health

            # Ensina o cérebro sobre TODOS os vizinhos atuais
            if agent.neighbors:
                share_delta = delta_e / len(agent.neighbors)
                for nid in agent.neighbors:
                    neighbor = self.agents.get(nid)
                    if neighbor:
                        agent.brain.learn(neighbor.genome, share_delta)

    def _update_ecosystem_metrics(self):
        """Atualiza métricas do ecossistema"""
        if not self.agents:
            return
        total_health = sum(agent.health for agent in self.agents.values())
        self.state.total_energy = total_health / len(self.agents)
        total_connections = sum(len(agent.neighbors) for agent in self.agents.values())
        self.state.structure_coherence = total_connections / (len(self.agents) * 6 + 1e-6)
        signal_values = self.field.signal_grid.flatten()
        self.state.signal_diversity = np.std(signal_values) / (np.mean(signal_values) + 1e-6)

    def get_render_data(self) -> Tuple[List[np.ndarray], List[float], List[List[int]]]:
        positions = []
        energies = []
        connections = []
        for agent in self.agents.values():
            if agent.health > 0:
                positions.append(agent.position.copy())
                energies.append(agent.health)
                connections.append(agent.neighbors.copy())
        return positions, energies, connections

    def inject_signal(self, position: np.ndarray, strength: float = 10.0):
        self._add_signal_source(position, strength)

    def clear_signals(self):
        self.signals.clear()
        # Sempre mantém o sinal central como base
        self._add_signal_source(np.array(FIELD_SIZE) // 2, 10.0)
