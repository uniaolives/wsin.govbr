from gui.unified_particle_system import UnifiedParticleSystem

class ConsciousnessVisualizer3D:
    def __init__(self):
        # Sistema de partículas
        self.particle_system = UnifiedParticleSystem(num_particles=120)

        # Controles de interface
        self.modes = ["MANDALA", "DNA", "HYPERCORE"]
        self.current_mode_index = 0

        # Conexão EEG (simulada)
        self.eeg_connected = False
        self.attention_level = 0.5
        self.meditation_level = 0.5

    def update_from_eeg(self, eeg_data):
        """Atualiza visualização baseada em dados EEG reais."""
        if eeg_data:
            # Assumindo que eeg_data tem atributos attention e meditation (0-100)
            self.attention_level = getattr(eeg_data, 'attention', 50) / 100.0
            self.meditation_level = getattr(eeg_data, 'meditation', 50) / 100.0

            # Muda modo baseado no estado mental
            if self.attention_level > 0.7:
                self.particle_system.set_mode("DNA")
            elif self.meditation_level > 0.7:
                self.particle_system.set_mode("HYPERCORE")
            else:
                self.particle_system.set_mode("MANDALA")

    def render_frame(self, dt):
        """Renderiza um frame da visualização."""
        # Atualiza sistema de partículas
        self.particle_system.update(dt)

        # Obtém dados para renderização
        data = self.particle_system.get_particle_data()

        # Em um sistema real, aqui chamaríamos funções OpenGL/WebGL
        # self.render_particles(data['positions'], data['colors'], data['sizes'])

        # Log de status para depuração no sandbox
        # print(f"Mode: {data['mode']} | Progress: {data['transition']:.2f}")

        return data

    def render_particles(self, positions, colors, sizes):
        """Placeholder para renderização real."""
        pass

    def render_hypercore_connections(self, positions):
        """Placeholder para renderizar arestas do Hecatonicosachoron."""
        pass

    def render_hud(self, hud_data):
        """Placeholder para interface de usuário."""
        pass

if __name__ == "__main__":
    # Teste rápido
    viz = ConsciousnessVisualizer3D()
    for i in range(10):
        viz.render_frame(0.1)
    print("Visualizer frame test completed.")
