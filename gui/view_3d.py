"""
Renderizador 3D para o ecossistema Bio-Gênese Cognitiva usando Pyglet.
"""

import numpy as np
from core.particle_system import CognitiveParticleEngine

# Tenta importar pyglet
try:
    import pyglet
    from pyglet.gl import *
    HAS_PYGLET = True
except Exception:
    HAS_PYGLET = False

if HAS_PYGLET:
    class CognitiveViewer(pyglet.window.Window):
        def __init__(self):
            super().__init__(1280, 720, "Bio-Gênese: Cognitiva", resizable=True)
            self.engine = CognitiveParticleEngine()
            self.camera_pos = [0, 0, 250]
            self.camera_rot = [30, 0]
            self.paused = False
            self.selected_info = ""
            self.label = pyglet.text.Label('', x=10, y=10, multiline=True, width=500)
            pyglet.clock.schedule_interval(self.update, 1/60.0)

        def update(self, dt):
            if not self.paused:
                self.engine.update(dt)

        def on_draw(self):
            self.clear()
            glViewport(0, 0, self.width, self.height)
            glMatrixMode(GL_PROJECTION); glLoadIdentity()
            gluPerspective(60, self.width/self.height, 0.1, 1000)
            glMatrixMode(GL_MODELVIEW); glLoadIdentity()
            glTranslatef(-self.camera_pos[0], -self.camera_pos[1], -self.camera_pos[2])
            glRotatef(self.camera_rot[0], 1, 0, 0)
            glRotatef(self.camera_rot[1], 0, 1, 0)
            glTranslatef(-50, -50, -50)

            # Agents
            pos, energy, conn, cognitive = self.engine.get_render_data()

            # Connections
            glBegin(GL_LINES)
            glColor4f(1, 1, 1, 0.1)
            # Line drawing logic... (simplified)
            glEnd()

            # Points
            glPointSize(5)
            glBegin(GL_POINTS)
            for i, p in enumerate(pos):
                cog = cognitive[i]
                if cog == "smart": glColor3f(0.1, 0.9, 0.2)
                elif cog == "confused": glColor3f(0.9, 0.1, 0.1)
                else: glColor3f(0.4, 0.4, 0.9)
                glVertex3f(*p)
            glEnd()

            # UI
            glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, self.width, 0, self.height, -1, 1)
            glMatrixMode(GL_MODELVIEW); glLoadIdentity()
            self.label.text = f"Sim Step: {self.engine.simulation_step}\nAgentes: {len(pos)}\nEnergia Média: {self.engine.state.total_energy:.3f}\nSucesso Médio: {self.engine.state.average_learning:.2f}"
            self.label.draw()

        def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
            if buttons & pyglet.window.mouse.LEFT:
                self.camera_rot[1] += dx * 0.5
                self.camera_rot[0] -= dy * 0.5

        def on_key_press(self, symbol, modifiers):
            if symbol == pyglet.window.key.SPACE: self.paused = not self.paused
            elif symbol == pyglet.window.key.R: self.engine = CognitiveParticleEngine()
            elif symbol == pyglet.window.key.I: self.engine.inject_signal(np.random.rand(3)*100, 20.0)

        def run(self): pyglet.app.run()
else:
    class CognitiveViewer:
        def __init__(self, *args, **kwargs): pass
        def run(self): pass
