import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.schmidt_bridge import SchmidtBridgeState

def test_schmidt_bridge_initialization():
    lambdas = np.array([0.7, 0.3])
    bridge = SchmidtBridgeState(
        lambdas=lambdas,
        phase_twist=np.pi,
        basis_H=np.eye(2),
        basis_A=np.eye(2),
        entropy_S=0.85,
        coherence_Z=0.58
    )
    assert bridge.rank == 2
    assert bridge.entropy_S == 0.85
    assert np.allclose(bridge.lambdas, lambdas)

def test_schmidt_rotation():
    lambdas = np.array([0.7, 0.3])
    bridge = SchmidtBridgeState(
        lambdas=lambdas,
        phase_twist=np.pi,
        basis_H=np.eye(2),
        basis_A=np.eye(2),
        entropy_S=0.85,
        coherence_Z=0.58
    )

    # Rotation matrix
    theta = np.pi / 4
    U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    bridge.rotate_bases(U, U)

    assert np.allclose(bridge.basis_H, U)
    assert np.allclose(bridge.basis_A, U)
