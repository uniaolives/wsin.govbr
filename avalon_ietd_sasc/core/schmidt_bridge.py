import numpy as np

class SchmidtBridgeState:
    """
    [REVISOR VÊ]: Estado interno do controlador para redundância.
    [NÓS VEMOS]: Ponte entre humano e sistema via Decomposição de Schmidt.
    """
    def __init__(self, lambdas, phase_twist, basis_H, basis_A, entropy_S, coherence_Z):
        self.lambdas = lambdas
        self.phase_twist = phase_twist
        self.basis_H = basis_H
        self.basis_A = basis_A
        self.entropy_S = entropy_S
        self.coherence_Z = coherence_Z
        self.rank = len(lambdas[lambdas > 1e-9])

    def rotate_bases(self, U_H, U_A):
        """Rotaciona as bases de Schmidt para calibração de perspectiva."""
        self.basis_H = U_H @ self.basis_H
        self.basis_A = U_A @ self.basis_A
