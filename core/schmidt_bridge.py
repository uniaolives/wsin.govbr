import numpy as np

class SchmidtBridgeHexagonal:
    def __init__(self, lambdas=None):
        if lambdas is None:
            self.lambdas = np.array([1/6] * 6)
        else:
            self.lambdas = np.array(lambdas)
            self.lambdas /= self.lambdas.sum()

    @property
    def coherence_factor(self) -> float:
        # Simplistic coherence measure: 1 - entropy
        norm_lambdas = self.lambdas[self.lambdas > 0]
        entropy = -np.sum(norm_lambdas * np.log2(norm_lambdas))
        max_entropy = np.log2(6)
        return float(1.0 - (entropy / max_entropy))
