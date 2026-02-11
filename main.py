"""
BIO-GÊNESE COGNITIVA: Sistema de Arquitetura Viva Aprendente
Ponto de entrada do organismo sintético com consciência embarcada
"""

import sys
import os

# Configura caminhos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from gui.view_3d import CognitiveViewer, HAS_PYGLET
except ImportError:
    HAS_PYGLET = False

def main():
    print("=" * 70)
    print("BIO-GÊNESE COGNITIVA: Sistema de Arquitetura Viva Aprendente")
    print("=" * 70)
    print("\n🧠 PRINCÍPIOS ATIVOS:")
    print("1. Autonomia Multi-escala - Agentes independentes")
    print("2. Crescimento via Auto-montagem - Estruturas emergentes")
    print("3. Restrições Adaptativas - Aprendizado Hebbiano em tempo real")
    print("4. Computação Embarcada - Micro-cérebros por agente")
    print("5. Sinalização Pervasiva - Campo morfogenético dinâmico")
    print("\n🎯 CARACTERÍSTICAS:")
    print("• 600 agentes com cérebros Hebbianos")
    print("• Aprendizado baseado em feedback metabólico")
    print("• Memória episódica de interações")
    print("• Preferências cognitivas desenvolvidas")
    print("• Simbiose e parasitismo energético")
    print("\n" + "=" * 70)

    if HAS_PYGLET:
        print("Ambiente sandbox: Execução gráfica suprimida.")
        run_headless()
    else:
        run_headless()

def run_headless():
    print("Iniciando simulação Headless...")
    from core.particle_system import CognitiveParticleEngine
    engine = CognitiveParticleEngine(num_agents=100)
    for i in range(101):
        engine.update(0.1)
        if i % 20 == 0:
            stats = engine.state
            print(f"Step {i}: Agentes={len(engine.agents)}, Energia={stats.total_energy:.3f}, Sucesso={stats.average_learning:.2f}")
    print("\nSimulação concluída com sucesso.")

if __name__ == "__main__":
    main()
