"""
HYPERCORE INTEGRATION DEMO v3.0
Demonstração da integração total Arkhé: Neural + QRL + Bio-Gênese + MCP
"""

import asyncio
import numpy as np
import time
from datetime import datetime

# 1. Componentes do Sistema
from core.bio_arkhe import BioAgent, ArkheGenome
from core.particle_system import BioGenesisEngine
from arkhe_qrl_integrated_system import QRLIntegratedBiofeedback
from arkhe_isomorphic_bridge import ArkheIsomorphicLab
from web_mcp_interface import WebMCPInterface

# 2. Visualização (Opcional, requer Pyglet)
try:
    from gui.view_3d import BioGenesisViewer
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False
    print("⚠️  Visualizador 3D não disponível (requer pyglet e PyOpenGL)")

async def run_hypercore_simulation():
    print("\n" + "🚀" * 30)
    print("      ARKHÉ HYPERCORE SYSTEM v3.0 - INITIALIZING")
    print("🚀" * 30 + "\n")

    # A. Inicializa Motores
    print("[1/4] Inicializando Motor de Bio-Gênese...")
    engine = BioGenesisEngine(num_agents=150)

    print("[2/4] Inicializando Loop de Biofeedback QRL...")
    qrl_system = QRLIntegratedBiofeedback(user_id="master_explorer")

    print("[3/4] Inicializando Laboratório Isomórfico...")
    lab = ArkheIsomorphicLab(user_id="master_explorer")

    print("[4/4] Inicializando Interface WebMCP...")
    mcp = WebMCPInterface(engine)

    # B. Configura Visualizador se disponível
    viewer = None
    if VISUALIZER_AVAILABLE:
        print("\n🎨 Iniciando Visualizador 3D...")
        viewer = BioGenesisViewer(engine, width=1280, height=720)

    print("\n" + "="*60)
    print("SISTEMA OPERACIONAL - PRESSIONE CTRL+C PARA ENCERRAR")
    print("="*60 + "\n")

    try:
        step = 0
        while True:
            t0 = time.time()

            # 1. Update Bio-Simulation
            engine.update(dt=0.1)

            # 2. Sync with QRL (Simulado aqui, no sistema real vem da câmera)
            if step % 20 == 0:
                stats = engine.get_stats()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Step {step} | "
                      f"Agentes: {stats['agents']} | Saúde Média: {stats['avg_health']}")

            # 3. Interação Isomórfica (Trigger aleatório para demo)
            if step == 100:
                print("\n🧪 [AUTO-TRIGGER] Designando molécula de suporte para a simulação...")
                await lab.consciousness_molecule_design_session(
                    target_experience="focused_flow",
                    verbal_intention="Estabilizar a harmonia da rede biogênica"
                )
                print("")

            # 4. Render Frame (Se viewer ativo)
            if viewer:
                viewer.on_draw()
                viewer.dispatch_events()

            # Controle de framerate
            elapsed = time.time() - t0
            await asyncio.sleep(max(0.01, 0.033 - elapsed))

            step += 1

    except KeyboardInterrupt:
        print("\n\n🛑 Encerrando Hypercore...")
    finally:
        print("✅ Sistema finalizado com sucesso.")

if __name__ == "__main__":
    asyncio.run(run_hypercore_simulation())
