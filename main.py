#!/usr/bin/env python3
"""
MAIN - Ponto de entrada do Bio-Gênese Cognitivo v3.0
Orquestra a simulação, a visualização e a prontidão WebMCP.
"""

import sys
import os
import numpy as np

# Adiciona diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    try:
        import numpy
        import pyglet
        return True
    except ImportError as e:
        print(f"Erro: Dependência faltando: {e}")
        return False

def run_headless(engine, steps=100):
    print(f"Executando simulação Headless ({steps} passos)...")
    for i in range(steps + 1):
        engine.update(0.1)
        if i % 25 == 0:
            stats = engine.get_stats()
            print(f"Passo {i}: Agentes={stats['agents']}, Tempo={stats['time']:.1f}, Vínculos={stats['bonds']}")
    print("Simulação concluída.")

def main():
    print("=" * 70)
    print("  BIO-GÊNESE COGNITIVA v3.0 - Sovereign Arkhe(n) Manifold")
    print("=" * 70)

    if not check_dependencies():
        print("Instale as dependências com: pip install numpy pyglet")
        sys.exit(1)

    from core.particle_system import BioGenesisEngine
    from gui.view_3d import BioGenesisViewer
    from web_mcp_interface import generate_webmcp_html

    # 1. Gera interface WebMCP para agentes AI
    with open("webmcp_interface.html", "w") as f:
        f.write(generate_webmcp_html())
    print("✓ Interface WebMCP gerada (webmcp_interface.html)")

    # 2. Inicializa o motor
    engine = BioGenesisEngine(num_agents=300)
    print("✓ População inicial gerada (3 tribos)")

    # 3. Decide modo de execução
    if os.environ.get('DISPLAY') or sys.platform == "darwin" or sys.platform == "win32":
        print("✓ Iniciando visualizador 3D...")
        try:
            from gui.view_3d import BioGenesisViewer
            # Como BioGenesisViewer cria sua própria engine por padrão,
            # aqui chamamos a função main do view_3d ou instanciamos.
            # No v3.0 de view_3d.py, main() chama pyglet.app.run().
            import pyglet
            window = BioGenesisViewer()
            pyglet.app.run()
        except Exception as e:
            print(f"Aviso: Falha ao iniciar interface gráfica: {e}")
            run_headless(engine)
    else:
        run_headless(engine)

if __name__ == "__main__":
    main()
