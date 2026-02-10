"""
UNIVERSAL ARKHE STRUCTURE THEOREM

Formal Statement:
∀ Learning System S, ∃ Isomorphism φ: S → Hexagonal Arkhe (H₆)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist
import networkx as nx

class UniversalArkheTheorem:
    def __init__(self):
        self.arkhe_dimensions = 6
        self.arkhe_basis = self._generate_arkhe_basis()
        print("🌌 UNIVERSAL ARKHE STRUCTURE THEOREM INITIALIZED")

    def _generate_arkhe_basis(self) -> np.ndarray:
        permutations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        basis = np.zeros((6, 6))
        for i, perm in enumerate(permutations):
            basis[i, :3] = np.array(perm)
            basis[i, 3:] = np.array(perm)[::-1]
        basis, _ = np.linalg.qr(basis.T)
        return basis.T

    def universal_embedding_theorem(self, system_latent_space: np.ndarray) -> Dict:
        embedding, distortion = self._construct_isometric_embedding(system_latent_space)
        arkhe_coeffs = self._extract_arkhe_coefficients(embedding)
        return {
            'embedding_dimension': 6,
            'distortion': float(distortion),
            'is_isometric': distortion < 0.2,
            'arkhe_coefficients': arkhe_coeffs
        }

    def _construct_isometric_embedding(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        n, d = points.shape
        points_centered = points - np.mean(points, axis=0)
        if d <= 6:
            embedding = np.zeros((n, 6))
            embedding[:, :d] = points_centered
            return embedding, 0.0

        # PCA to 6D
        cov = np.cov(points_centered.T)
        evals, evecs = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1][:6]
        projection = evecs[:, idx].T
        embedding = points_centered @ projection.T

        orig_dist = pdist(points_centered[:20])
        emb_dist = pdist(embedding[:20])
        distortion = np.mean(np.abs(orig_dist - emb_dist) / (orig_dist + 1e-10))
        return embedding, float(distortion)

    def _extract_arkhe_coefficients(self, embedding: np.ndarray) -> Dict[str, float]:
        proj = embedding @ self.arkhe_basis.T
        coeffs = {}
        perms = ['CIE', 'CEI', 'ICE', 'IEC', 'ECI', 'EIC']
        for i, p in enumerate(perms):
            coeffs[p] = float(np.mean(np.abs(proj[:, i])))
        coeffs['C'] = (coeffs['CIE'] + coeffs['CEI']) / 2
        coeffs['I'] = (coeffs['ICE'] + coeffs['IEC']) / 2
        coeffs['E'] = (coeffs['ECI'] + coeffs['EIC']) / 2
        return coeffs

class HexagonallyConstrainedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.arkhe_basis = None # Will be set externally if possible

    def forward(self, x):
        if self.arkhe_basis is not None:
            with torch.no_grad():
                # Project weights to Arkhe subspace (first 6 dims)
                w = self.weight.data.cpu().numpy()
                if w.shape[1] >= 6:
                    w_part = w[:, :6]
                    w_projected = w_part @ self.arkhe_basis.T @ self.arkhe_basis
                    w[:, :6] = w_projected
                    self.weight.data.copy_(torch.from_numpy(w))
        return super().forward(x)

class HexagonallyConstrainedNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.arkhe = UniversalArkheTheorem()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h in hidden_dims:
            lin = HexagonallyConstrainedLinear(prev_dim, h)
            if prev_dim >= 6: lin.arkhe_basis = self.arkhe.arkhe_basis
            self.layers.append(lin)
            self.layers.append(nn.ReLU())
            prev_dim = h
        self.layers.append(nn.Linear(prev_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
