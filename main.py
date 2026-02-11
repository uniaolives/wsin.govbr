"""
ARKHÉ MASTER SYSTEM v3.0 - Entry Point
Sistema Integrado de Biofeedback Quântico, Bio-Gênese e Design Molecular.
"""

import sys
import os
import asyncio

def check_dependencies():
    """Verifica se todas as dependências críticas estão instaladas."""
    print("🔍 Verificando ecossistema Arkhé...")

    missing = []

    # Lista de pacotes necessários
    packages = {
        'numpy': 'numpy',
        'torch': 'torch',
        'qiskit': 'qiskit',
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'sklearn': 'scikit-learn',
        'scipy': 'scipy',
        'pyglet': 'pyglet',
        'OpenGL': 'PyOpenGL'
    }

    for module, package in packages.items():
        try:
            __import__(module)
            print(f"  [OK] {module}")
        except ImportError:
            print(f"  [ERRO] {module} não encontrado.")
            missing.append(package)

    if missing:
        print("\n❌ Faltam dependências. Por favor, execute:")
        print(f"pip install {' '.join(missing)}")
        return False

    print("✅ Todas as dependências verificadas.\n")
    return True

async def start_system():
    """Inicia o sistema Arkhé."""
    if not check_dependencies():
        return

    print("="*60)
    print("             ARKHÉ MASTER SYSTEM v3.0")
    print("         'The Verbe becomes Molecule and Soul'")
    print("="*60)

    print("\nEscolha o módulo para iniciar:")
    print("1. Hypercore Demo (Sistema Total Integrado)")
    print("2. Biofeedback Quântico (Neural + QRL)")
    print("3. Laboratório Isomórfico (Design Molecular)")
    print("4. Bio-Gênese Simulação (Apenas Simulação)")
    print("5. Sair")

    try:
        choice = input("\nSeleção > ")

        if choice == '1':
            from hypercore_demo import run_hypercore_simulation
            await run_hypercore_simulation()
        elif choice == '2':
            from arkhe_qrl_integrated_system import main_qrl
            await main_qrl()
        elif choice == '3':
            from arkhe_isomorphic_bridge import arkhe_isomorphic_demo
            await arkhe_isomorphic_demo()
        elif choice == '4':
            print("Iniciando simulação 3D...")
            from core.particle_system import BioGenesisEngine
            from gui.view_3d import BioGenesisViewer
            import pyglet

            engine = BioGenesisEngine(num_agents=200)
            viewer = BioGenesisViewer(engine)
            pyglet.app.run()
        else:
            print("Encerrando.")

    except Exception as e:
        print(f"\n❌ Erro crítico no sistema: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(start_system())
