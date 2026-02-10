import asyncio
import numpy as np
import datetime
from arkhe_qrl_integrated_system import QRLIntegratedBiofeedback
from knn_emotion_enhancer import KNNEnhancedFacialBiofeedback
from arkhe_isomorphic_bridge import ArkheIsomorphicLab

async def test_full_pipeline():
    print("Starting Arkhé Full Integration Test...")

    # 1. Test KNN Enhanced Biofeedback
    print("\n[1/4] Testing KNN Biofeedback...")
    knn_system = KNNEnhancedFacialBiofeedback(user_id="test_user")
    # Simulate a pattern
    analysis = {
        'face_detected': True,
        'landmarks': type('Landmarks', (), {'landmark': [type('Point', (), {'x': 0.1, 'y': 0.1, 'z': 0.1}) for _ in range(468)]})(),
        'emotion': 'happy',
        'valence': 0.8,
        'arousal': 0.6,
        'facial_asymmetry': 0.05,
        'timestamp': datetime.datetime.now(),
        'microexpressions': []
    }
    await knn_system.process_emotional_state(analysis)
    print("KNN Biofeedback OK")

    # 2. Test Neural + QRL Integrated System
    print("\n[2/4] Testing Neural + QRL Integration...")
    qrl_system = QRLIntegratedBiofeedback(user_id="test_user")
    for i in range(6): # Force at least one sequence addition
        await qrl_system.process_emotional_state(analysis)
    print("Neural + QRL Integration OK")

    # 3. Test Isomorphic Bridge
    print("\n[3/4] Testing Isomorphic Bridge...")
    lab = ArkheIsomorphicLab(user_id="test_user")
    results = await lab.consciousness_molecule_design_session(
        target_experience="focused_flow",
        verbal_intention="Estou totalmente focado e presente"
    )
    print(f"Isomorphic Bridge OK - Molecule: {results['molecule'].drug_name}")

    # 4. Final verification of generated reports
    print("\n[4/4] Verifying generated reports...")
    import os
    reports = os.listdir("reports")
    print(f"Reports generated: {reports}")

    print("\nAll integration tests PASSED!")

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
