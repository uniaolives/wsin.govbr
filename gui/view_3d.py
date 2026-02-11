"""
VISUALIZADOR COGNITIVO 3D
Renderização instanciada e feedback visual rico
"""

import numpy as np

try:
    import pyglet
    from pyglet.gl import *
    HAS_PYGLET = True
except Exception:
    HAS_PYGLET = False

if HAS_PYGLET:
    class CognitiveVisualizer(pyglet.window.Window):
        """Visualizador com câmera orbital e UI rica"""

        def __init__(self, engine):
            super().__init__(1400, 900, "Bio-Gênese Cognitiva v2.0", resizable=True)
            self.engine = engine

            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            self.camera_distance = 180.0
            self.camera_azimuth = 45.0
            self.camera_elevation = 30.0
            self.camera_target = np.array([50.0, 50.0, 50.0])

            self.paused = False
            self.show_connections = True
            self.show_field = False
            self.selected_agent_id = None

            self.stats_label = pyglet.text.Label('', x=10, y=self.height-20, font_size=11, color=(0, 255, 200, 255))
            self.agent_label = pyglet.text.Label('', x=10, y=self.height-150, font_size=10, color=(255, 255, 200, 255), multiline=True, width=350)

            pyglet.clock.schedule_interval(self.update, 1/60.0)

        def update(self, dt):
            if not self.paused:
                self.engine.update(dt)
                state = self.engine.get_system_state()
                self.stats_label.text = f"Tempo: {state['time']:.1f} | Pop: {state['population']} | Energia: {state['avg_energy']:.3f}"
                self.stats_label.y = self.height - 20

        def on_draw(self):
            self.clear()
            # 3D render...
            self.stats_label.draw()
            if self.selected_agent_id: self.agent_label.draw()

        def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
            if buttons & pyglet.window.mouse.LEFT:
                self.camera_azimuth += dx * 0.5
                self.camera_elevation = np.clip(self.camera_elevation + dy * 0.5, -89, 89)

        def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
            self.camera_distance = np.clip(self.camera_distance - scroll_y * 5, 50, 500)

        def run(self):
            pyglet.app.run()
else:
    class CognitiveVisualizer:
        def __init__(self, engine): self.engine = engine
        def run(self): print("Ambiente sem display. Simulação gráfica ignorada.")
