"""
VIEW_3D - Visualizador Pyglet Simplificado e Funcional
"""

import numpy as np
import sys
import os

# Tenta importar pyglet com segurança
HAS_PYGLET = False
try:
    # Apenas tenta importar se houver display ou se não estiver no sandbox problemático
    if os.environ.get('DISPLAY'):
        import pyglet
        from pyglet.gl import *
        HAS_PYGLET = True
except Exception:
    HAS_PYGLET = False

# Adiciona diretório pai ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.particle_system import BioGenesisEngine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

if HAS_PYGLET:
    class BioGenesisViewer(pyglet.window.Window):
        def __init__(self, width=1200, height=800):
            super().__init__(width, height, "Bio-Gênese Cognitiva v3.0", resizable=True)

            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_POINT_SMOOTH)

            if HAS_ENGINE:
                self.engine = BioGenesisEngine(num_agents=250)
            else:
                self.engine = None

            self.camera = {'dist': 180, 'rot_x': 30, 'rot_y': 45}
            self.paused = False

            self.stats_label = pyglet.text.Label('', x=10, y=height-30,
                                               font_size=11, color=(0, 255, 200, 255))

            pyglet.clock.schedule_interval(self.update, 1/60.0)

        def update(self, dt):
            if not self.paused and self.engine:
                self.engine.update(dt)
                stats = self.engine.get_stats()
                self.stats_label.text = (f"Agentes: {stats['agents']} | "
                                       f"Tempo: {stats['time']:.1f} | "
                                       f"Vínculos: {stats['bonds']}")

        def on_draw(self):
            self.clear()
            glViewport(0, 0, self.width, self.height)
            glMatrixMode(GL_PROJECTION); glLoadIdentity()
            gluPerspective(60, self.width/self.height, 1, 1000)
            glMatrixMode(GL_MODELVIEW); glLoadIdentity()
            glTranslatef(0, 0, -self.camera['dist'])
            glRotatef(self.camera['rot_x'], 1, 0, 0)
            glRotatef(self.camera['rot_y'], 0, 1, 0)
            glTranslatef(-50, -50, -50)

            if self.engine:
                positions, healths, connections, profiles = self.engine.get_render_data()
                glBegin(GL_LINES)
                glColor4f(0.4, 0.7, 1.0, 0.25)
                # ... connections ...
                glEnd()
                glPointSize(7.0)
                glBegin(GL_POINTS)
                for i, (pos, health, profile) in enumerate(zip(positions, healths, profiles)):
                    if profile == "Especialista": r, g, b = 0.0, 1.0, 0.2
                    elif profile == "Aprendiz": r, g, b = 1.0, 0.9, 0.0
                    elif profile == "Cauteloso": r, g, b = 1.0, 0.4, 0.0
                    else: r, g, b = 0.3, 0.5, 1.0
                    factor = 0.4 + health * 0.6
                    glColor3f(r * factor, g * factor, b * factor)
                    glVertex3f(*pos)
                glEnd()

            glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, self.width, 0, self.height, -1, 1)
            glMatrixMode(GL_MODELVIEW); glLoadIdentity()
            self.stats_label.draw()

        def run(self):
            pyglet.app.run()
else:
    class BioGenesisViewer:
        def __init__(self, *args, **kwargs):
            pass
        def run(self):
            print("Visualizador gráfico não disponível.")

def main():
    if HAS_PYGLET:
        print("Iniciando visualizador...")
        window = BioGenesisViewer()
        window.run()
    else:
        print("Pyglet não disponível ou sem DISPLAY.")

if __name__ == "__main__":
    main()
