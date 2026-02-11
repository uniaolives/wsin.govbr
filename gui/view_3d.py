"""
VIEW 3D v3.0 - Visualização OpenGL para Bio-Gênese
Renderiza agentes, campos e conexões sociais.
"""

import pyglet
from pyglet.gl import *
import numpy as np
from typing import List, Dict, Tuple

class BioGenesisViewer(pyglet.window.Window):
    """
    Visualizador 3D para a simulação Bio-Gênese.
    """

    def __init__(self, engine, width=1024, height=768):
        super().__init__(width=width, height=height, caption="Arkhé Bio-Gênese v3.0", resizable=True)
        self.engine = engine
        self.batch = pyglet.graphics.Batch()

        # Configuração da câmera
        self.rotation = [0, 0]
        self.zoom = -150

        # Setup OpenGL
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def on_resize(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / float(height), 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)
        return pyglet.event.EVENT_HANDLED

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.rotation[0] += dx * 0.5
        self.rotation[1] -= dy * 0.5

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.zoom += scroll_y * 5

    def draw_cube_outline(self, size):
        glBegin(GL_LINES)
        glColor4f(1, 1, 1, 0.2)
        # Borda inferior
        glVertex3f(0, 0, 0); glVertex3f(size, 0, 0)
        glVertex3f(size, 0, 0); glVertex3f(size, 0, size)
        glVertex3f(size, 0, size); glVertex3f(0, 0, size)
        glVertex3f(0, 0, size); glVertex3f(0, 0, 0)
        # Borda superior
        glVertex3f(0, size, 0); glVertex3f(size, size, 0)
        glVertex3f(size, size, 0); glVertex3f(size, size, size)
        glVertex3f(size, size, size); glVertex3f(0, size, size)
        glVertex3f(0, size, size); glVertex3f(0, size, 0)
        # Colunas
        glVertex3f(0, 0, 0); glVertex3f(0, size, 0)
        glVertex3f(size, 0, 0); glVertex3f(size, size, 0)
        glVertex3f(size, 0, size); glVertex3f(size, size, size)
        glVertex3f(0, 0, size); glVertex3f(0, size, size)
        glEnd()

    def draw_agents(self, positions, healths):
        glPointSize(5.0)
        glBegin(GL_POINTS)
        for i, pos in enumerate(positions):
            h = healths[i]
            # Cor baseada na saúde (Verde -> Vermelho)
            glColor4f(1.0 - h, h, 0.5, 0.8)
            glVertex3f(pos[0], pos[1], pos[2])
        glEnd()

    def draw_connections(self, positions, connection_map):
        glBegin(GL_LINES)
        glColor4f(0.4, 0.6, 1.0, 0.15) # Azul suave transparente
        for i, neighbors in enumerate(connection_map):
            if i >= len(positions): continue
            p1 = positions[i]
            for neighbor_id in neighbors:
                # Aqui simplificamos: o id do vizinho pode não ser o índice no array positions
                # Mas para a visualização rápida, tentamos mapear se possível
                # No engine v3, guardamos os IDs reais.
                pass
        glEnd()

    def on_draw(self):
        self.clear()
        glLoadIdentity()
        glTranslatef(0, 0, self.zoom)
        glRotatef(self.rotation[1], 1, 0, 0)
        glRotatef(self.rotation[0], 0, 1, 0)

        # Centraliza o cubo
        glTranslatef(-50, -50, -50)

        self.draw_cube_outline(100)

        # Get data from engine
        positions, healths, connection_map, _ = self.engine.get_render_data()

        # Desenha conexões
        glBegin(GL_LINES)
        glColor4f(0.5, 0.7, 1.0, 0.1)
        # Otimização: desenha apenas algumas conexões para manter performance
        for i, pos in enumerate(positions):
            neighbors = connection_map[i]
            # Como render_data retorna listas ordenadas, precisamos garantir o mapeamento de ID
            # Para o viewer v3.0, desenhamos linhas baseadas na proximidade se houver conexão
            for other_idx in range(len(positions)):
                if other_idx == i: continue
                dist = np.linalg.norm(np.array(pos) - np.array(positions[other_idx]))
                if dist < 4.0:
                    glVertex3f(pos[0], pos[1], pos[2])
                    glVertex3f(positions[other_idx][0], positions[other_idx][1], positions[other_idx][2])
        glEnd()

        self.draw_agents(positions, healths)

    def update(self, dt):
        self.engine.update(dt)
