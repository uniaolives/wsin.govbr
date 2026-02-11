"""
Ponto de entrada do Sistema Bio-Gênese
"""

import sys
import os

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Tenta importar o visualizador
try:
    from gui.view_3d import BioGenesisViewer
    HAS_GUI = True
except ImportError as e:
    print(f"Aviso: Não foi possível carregar o visualizador GUI: {e}")
    HAS_GUI = False

def main():
    print("=" * 60)
    print("BIO-GÊNESE: Sistema de Arquitetura Viva")
    print("=" * 60)
    print("\nPrincípios Ativos:")
    print("1. Autonomia Multi-escala - Agentes independentes")
    print("2. Crescimento via Auto-montagem - Estruturas emergentes")
    print("3. Restrições Adaptativas - Regras em tempo real")
    print("4. Computação Embarcada - Lógica distribuída")
    print("5. Sinalização Pervasiva - Campo morfogenético")
    print("\n" + "=" * 60)

    if HAS_GUI:
        print("Iniciando visualizador 3D...")
        # viewer = BioGenesisViewer()
        # viewer.run()
        print("Ambiente de demonstração: Execução de GUI suprimida para o sandbox.")
        # Simula execução em modo console
        run_headless()
    else:
        run_headless()

def run_headless():
    print("Executando em modo Headless (Simulação)...")
    from core.particle_system import BioParticleEngine
    engine = BioParticleEngine(num_agents=100)
    for i in range(100):
        engine.update(0.1)
        if i % 10 == 0:
            stats = engine.state
            print(f"Step {i}: Agentes={len(engine.agents)}, Energia={stats.total_energy:.3f}, Coerência={stats.structure_coherence:.3f}")
    print("Simulação concluída.")

if __name__ == "__main__":
    main()
