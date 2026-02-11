import math
import numpy as np

def get_mandala_pos(index, total_particles, time_pulse):
    """
    Gera posições para um padrão de Mandala (Circular/Geométrico).
    """
    # Camadas circulares
    layer_size = 30
    layer = index // layer_size
    idx_in_layer = index % layer_size

    radius = (layer + 1) * 2.0
    angle = (idx_in_layer / layer_size) * 2 * math.pi + time_pulse * 0.2

    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    z = 0.5 * math.sin(time_pulse + index * 0.2)

    return np.array([x, y, z])

def get_dna_pos(index, total_particles, time_pulse):
    """
    Gera posições para um padrão de Dupla Hélice (DNA).
    """
    # Alterna entre as duas fitas
    strand = 1 if index % 2 == 0 else -1

    # Altura ao longo do eixo Z
    z = (index / total_particles) * 20.0 - 10.0

    # Rotação helicoidal
    angle = (index / 10.0) * math.pi + time_pulse * 0.5

    radius = 3.0
    x = strand * radius * math.cos(angle)
    y = strand * radius * math.sin(angle)

    return np.array([x, y, z])

def get_hypercore_pos(index, total_particles, time_pulse, rotation_4d=True):
    """
    Gera posições 3D projetadas a partir do Hecatonicosachoron 4D.
    """
    # 1. GERAR VÉRTICES DO HECATONICOSACHORON (120 células)
    vertices_4d = []

    phi = (1 + math.sqrt(5)) / 2  # Proporção áurea

    # Conjunto 1: 8 vértices (±1, 0, 0, 0) e permutações
    ones = [1, -1]
    for i in range(4):
        for sign in ones:
            v = [0.0, 0.0, 0.0, 0.0]
            v[i] = float(sign)
            vertices_4d.append(v)

    # Conjunto 2: 16 vértices (±0.5, ±0.5, ±0.5, ±0.5)
    half = 0.5
    for sx in ones:
        for sy in ones:
            for sz in ones:
                for sw in ones:
                    neg_count = (sx < 0) + (sy < 0) + (sz < 0) + (sw < 0)
                    if neg_count % 2 == 0:
                        vertices_4d.append([sx*half, sy*half, sz*half, sw*half])

    # Conjunto 3: Adicionando mais alguns vértices para chegar perto de 120
    # (±0.5*phi, ±0.5, ±0.5/phi, 0) e permutações
    # Simplificado para completar a lista
    while len(vertices_4d) < 120:
        v = np.random.randn(4)
        v = v / np.linalg.norm(v)
        vertices_4d.append(v.tolist())

    # 2. SELEÇÃO DO VÉRTICE BASE
    vertex_idx = index % len(vertices_4d)
    point_4d = np.array(vertices_4d[vertex_idx])

    # 3. ROTAÇÃO 4D
    if rotation_4d:
        angle1 = time_pulse * 0.2  # Rotação XY
        angle2 = time_pulse * 0.15  # Rotação ZW

        cos1, sin1 = math.cos(angle1), math.sin(angle1)
        cos2, sin2 = math.cos(angle2), math.sin(angle2)

        x, y, z, w = point_4d
        x_new = x * cos1 - y * sin1
        y_new = x * sin1 + y * cos1
        z_new = z * cos2 - w * sin2
        w_new = z * sin2 + w * cos2

        point_4d = np.array([x_new, y_new, z_new, w_new])

    # 4. PROJEÇÃO ESTEREOGRÁFICA (4D → 3D)
    x, y, z, w = point_4d
    if abs(1 - w) < 0.001:
        w = 0.999

    scale = 1.0 / (1.0 - w)
    x_3d = x * scale
    y_3d = y * scale
    z_3d = z * scale

    # 5. ANIMAÇÃO ADICIONAL
    pulse = 1.0 + 0.1 * math.sin(time_pulse * 3 + index * 0.1)

    return np.array([x_3d * pulse, y_3d * pulse, z_3d * pulse])

class UnifiedParticleSystem:
    """
    Sistema de partículas que representa estados de consciência.

    Modos:
    - MANDALA: Ordem/Proteção (estado base)
    - DNA: Vida/Evolução (estado dinâmico)
    - HYPERCORE: Consciência 4D/Transmissão (estado elevado)
    """

    # Mapeamento direto entre modos visuais e estados Arkhe
    ARKHE_MODE_MAP = {
        "MANDALA": {"C": 0.6, "I": 0.2, "E": 0.1, "F": 0.1},  # Dominância Química
        "DNA": {"C": 0.2, "I": 0.5, "E": 0.2, "F": 0.1},      # Dominância Informacional
        "HYPERCORE": {"C": 0.1, "I": 0.3, "E": 0.3, "F": 0.3}  # Equilíbrio 4D
    }

    def __init__(self, num_particles=120):
        self.particles = []
        self.time = 0.0
        self.current_mode = "MANDALA"
        self.target_mode = "MANDALA"
        self.transition_progress = 0.0
        self.transition_speed = 0.02

        for i in range(num_particles):
            self.particles.append({
                'index': i,
                'pos': np.array([0.0, 0.0, 0.0]),
                'target_pos': np.array([0.0, 0.0, 0.0]),
                'color': [1.0, 1.0, 1.0, 1.0],
                'size': 1.0,
                'energy': 0.5 + 0.5 * math.sin(i * 0.1)
            })

    def set_mode(self, new_mode):
        if new_mode in ["MANDALA", "DNA", "HYPERCORE"]:
            if new_mode != self.target_mode:
                self.target_mode = new_mode
                self.transition_progress = 0.0

    def update(self, dt):
        self.time += dt

        if self.current_mode != self.target_mode:
            self.transition_progress += self.transition_speed
            if self.transition_progress >= 1.0:
                self.transition_progress = 1.0
                self.current_mode = self.target_mode

        for p in self.particles:
            idx = p['index']

            if self.target_mode == "MANDALA":
                target = get_mandala_pos(idx, len(self.particles), self.time)
            elif self.target_mode == "DNA":
                target = get_dna_pos(idx, len(self.particles), self.time)
            else:
                target = get_hypercore_pos(idx, len(self.particles), self.time)

            if self.transition_progress < 1.0:
                if self.current_mode == "MANDALA":
                    current = get_mandala_pos(idx, len(self.particles), self.time)
                elif self.current_mode == "DNA":
                    current = get_dna_pos(idx, len(self.particles), self.time)
                else:
                    current = get_hypercore_pos(idx, len(self.particles), self.time)

                t = self.transition_progress
                smooth_t = t * t * (3 - 2 * t)
                p['target_pos'] = current * (1 - smooth_t) + target * smooth_t
            else:
                p['target_pos'] = target

            p['pos'] = p['pos'] * 0.9 + p['target_pos'] * 0.1
            self.update_particle_color(p)
            noise = np.random.normal(0, 0.01, 3)
            p['pos'] += noise

    def update_particle_color(self, p):
        if self.current_mode == "MANDALA":
            hue, saturation, value = 0.12, 0.8, 0.7 + 0.3 * p['energy']
        elif self.current_mode == "DNA":
            hue, saturation, value = 0.5, 0.9, 0.6 + 0.4 * math.sin(self.time + p['index'] * 0.1)
        else:
            hue, saturation, value = 0.8, 0.7, 0.5 + 0.5 * math.sin(self.time * 2 + p['index'] * 0.05)

        p['color'] = self.hsv_to_rgb(hue, saturation, value)

    def hsv_to_rgb(self, h, s, v):
        i = math.floor(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        if i % 6 == 0: r, g, b = v, t, p
        elif i % 6 == 1: r, g, b = q, v, p
        elif i % 6 == 2: r, g, b = p, v, t
        elif i % 6 == 3: r, g, b = p, q, v
        elif i % 6 == 4: r, g, b = t, p, v
        elif i % 6 == 5: r, g, b = v, p, q

        return [r, g, b, 1.0]

    def get_particle_data(self):
        return {
            'positions': [p['pos'].tolist() for p in self.particles],
            'colors': [p['color'] for p in self.particles],
            'sizes': [p['size'] for p in self.particles],
            'mode': self.current_mode,
            'transition': self.transition_progress
        }
