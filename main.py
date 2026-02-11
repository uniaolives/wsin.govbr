#!/usr/bin/env python3
"""
BIO-GÊNESE COGNITIVA v2.0
Ponto de entrada do sistema de vida artificial com cognição embarcada.
"""

import sys
import os

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    try:
        import numpy
        import scipy
        return True
    except ImportError as e:
        print(f"Erro: Dependência faltando: {e}")
        return False

def run_headless(engine, steps=100):
    print(f"Executando simulação Headless ({steps} passos)...")
    for i in range(steps + 1):
        engine.update(0.1)
        if i % 20 == 0:
            state = engine.get_system_state()
            print(f"Passo {i}: Pop={state['population']}, Energia={state['avg_energy']:.3f}, Vínculos={state['bonds']}")
    print("Simulação concluída.")

def main():
    print("=" * 70)
    print("  BIO-GÊNESE COGNITIVA v2.0")
    print("  Sistema de Vida Artificial com Cognição Embarcada")
    print("=" * 70)

    if not check_dependencies():
        sys.exit(1)

    from core.particle_system import BioGenesisEngine
    from gui.view_3d import CognitiveVisualizer, HAS_PYGLET

    engine = BioGenesisEngine(num_agents=400)

    if HAS_PYGLET and os.environ.get('DISPLAY'):
        print("Iniciando visualizador 3D...")
        window = CognitiveVisualizer(engine)
        window.run()
    else:
        run_headless(engine)

if __name__ == "__main__":
    main()
