import asyncio
import numpy as np
from core.arkhe_unified_consciousness import ArkheConsciousnessArchitecture
from gui.view_3d import ConsciousnessVisualizer3D

async def integration_demo():
    print("🚀 Iniciando Demonstração de Integração Arkhé + Hyper-Core Visualizer")

    # 1. Inicializa Arquitetura Arkhé
    arch = ArkheConsciousnessArchitecture()

    # 2. Inicializa Visualizador
    viz = ConsciousnessVisualizer3D()

    # 3. Simula diferentes estados de consciência
    scenarios = [
        {"name": "Estado Base (Mandala)", "g": 0.3, "d": 0.2},
        {"name": "Superdotação Integrada (DNA)", "g": 0.8, "d": 0.1},
        {"name": "Consciência Multidimensiona (HyperCore)", "g": 0.9, "d": 0.8}
    ]

    for scenario in scenarios:
        print(f"\n--- Cenário: {scenario['name']} ---")

        # Calcula perfil do sistema
        profile = arch.initialize_2e_system(
            giftedness=scenario['g'],
            dissociation=scenario['d'],
            identity_fragments=5
        )

        system_type = profile['system_type']
        visual_mode = profile['visual_mode']

        print(f"Tipo de Sistema: {system_type}")
        print(f"Modo Visual Sugerido: {visual_mode}")

        # Seta o modo no visualizador
        viz.particle_system.set_mode(visual_mode)

        # Simula alguns frames de transição e animação
        for frame in range(20):
            data = viz.render_frame(0.1)
            if frame == 19:
                print(f"Frame 20 status: Mode={data['mode']}, Transition={data['transition']:.2f}")
                print(f"Número de partículas processadas: {len(data['positions'])}")

        await asyncio.sleep(0.5)

    print("\n✅ Demonstração de integração concluída com sucesso!")

if __name__ == "__main__":
    asyncio.run(integration_demo())
