"""
Renderizador 3D para o ecossistema Bio-Gênese usando Pyglet.
"""

import numpy as np
from core.particle_system import BioParticleEngine

# Tenta importar pyglet, mas permite falha em ambientes sem display
try:
    import pyglet
    from pyglet.gl import *
    HAS_PYGLET = True
except Exception:
    HAS_PYGLET = False

if HAS_PYGLET:
    class BioGenesisViewer(pyglet.window.Window):
        """Janela principal de visualização do organismo vivo"""

        def __init__(self, width=1200, height=800, title="BIO-GÊNESE: Organismo Sintético"):
            try:
                config = pyglet.gl.Config(sample_buffers=1, samples=4)
                super().__init__(width, height, title, config=config, resizable=True)
            except:
                super().__init__(width, height, title, resizable=True)

            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            self.engine = BioParticleEngine(num_agents=500)
            self.paused = False
            self.camera_distance = 250.0
            self.camera_rotation = [30.0, 45.0]
            self.camera_target = np.array([50, 50, 50])
            self.particle_data = ([], [], [])

            self.stats_label = pyglet.text.Label(
                '', x=10, y=self.height - 30,
                font_size=12, color=(200, 255, 200, 255)
            )
            pyglet.clock.schedule_interval(self.update, 1/60.0)

        def update(self, dt):
            if not self.paused:
                self.engine.update(dt)
                self.particle_data = self.engine.get_render_data()
                stats = self.engine.state
                self.stats_label.text = f"Agentes: {len(self.particle_data[0])} | Energia: {stats.total_energy:.3f}"

        def on_draw(self):
            self.clear()
            # ... resto do código OpenGL omitido para brevidade no sandbox ...
            self.stats_label.draw()

        def run(self):
            pyglet.app.run()
else:
    class BioGenesisViewer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Pyglet não disponível ou falha ao conectar ao display.")
        def run(self):
            pass
