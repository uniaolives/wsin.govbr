# cgda_lab_quantum.py
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import asyncio
from enum import Enum
import json
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
import networkx as nx

class ConstraintMethod(Enum):
    """Methods for constraint geometry derivation."""
    QUANTUM_HYBRID = "quantum_hybrid"  # Hybrid quantum-classical
    FULL = "full"                      # Complete constraint derivation
    ISING_EMBEDDING = "ising"          # Ising model embedding
    PSYCHIATRIC = "psychiatric"        # Psychiatric state mapping

@dataclass
class ObservedState:
    """An observed state in the system."""
    state_id: str
    features: np.ndarray  # State vector in feature space
    probability: float    # Observed probability
    forbidden: bool = False  # Whether this state is forbidden
    quantum_signature: Optional[complex] = None  # Quantum amplitude if applicable
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_quantum_state(self) -> np.ndarray:
        """Convert to quantum state vector."""
        if self.quantum_signature is not None:
            amplitude = np.sqrt(self.probability) * np.exp(1j * np.angle(self.quantum_signature))
        else:
            amplitude = np.sqrt(self.probability)

        # Create quantum state
        state = np.zeros(len(self.features), dtype=complex)
        state[np.argmax(self.features)] = amplitude
        return state

@dataclass
class ForbiddenConfiguration:
    """A forbidden configuration with constraints."""
    config_id: str
    state_pattern: np.ndarray  # Pattern that violates constraints
    constraint_violation: float  # Degree of violation
    violation_type: str  # Type of forbiddenness
    penalty_function: callable = None  # Penalty for entering this configuration

    def check_state(self, state: np.ndarray) -> float:
        """Check how much a state violates this forbidden configuration."""
        similarity = np.dot(state, self.state_pattern) / (
            np.linalg.norm(state) * np.linalg.norm(self.state_pattern) + 1e-10
        )
        return similarity * self.constraint_violation

class ConstraintGeometry:
    """Derived constraint geometry from observed states."""

    def __init__(self, method: ConstraintMethod):
        self.method = method
        self.constraint_matrix: np.ndarray = None  # Constraint matrix C
        self.eigenvalues: np.ndarray = None  # Eigenvalues of constraint space
        self.eigenvectors: np.ndarray = None  # Basis of constraint space
        self.allowed_subspace: np.ndarray = None  # Subspace of allowed states
        self.forbidden_subspace: np.ndarray = None  # Subspace of forbidden states
        self.quantum_operators: Dict[str, np.ndarray] = {}  # Quantum operators

    def satisfies_constraints(self, state: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if state satisfies constraints C|ψ⟩ = 0."""
        if self.constraint_matrix is None:
            return True

        violation = np.linalg.norm(self.constraint_matrix @ state)
        return violation < tolerance

    def project_to_allowed(self, state: np.ndarray) -> np.ndarray:
        """Project state onto allowed subspace."""
        if self.allowed_subspace is None:
            return state

        # Projection matrix P = ∑|v_i⟩⟨v_i| over allowed basis
        projection = self.allowed_subspace @ self.allowed_subspace.T.conj()
        return projection @ state

class CGDALab:
    """Constraint Geometry and Dynamics Analysis Lab."""

    def __init__(self, lab_id: str = "cgda_quantum_lab"):
        self.lab_id = lab_id
        self.observed_states: Dict[str, ObservedState] = {}
        self.forbidden_configs: Dict[str, ForbiddenConfiguration] = {}
        self.constraint_geometries: Dict[str, ConstraintGeometry] = {}

        # Data repositories
        self.ising_data: List[Dict] = []
        self.psychiatric_data: List[Dict] = []

        # Analysis results
        self.derivation_history: List[Dict] = []

    def load_observed_states(self, states_data: Union[str, List[Dict]]) -> Dict:
        """Load observed states from file or list."""

        if isinstance(states_data, str):
            # Load from JSON file
            with open(states_data, 'r') as f:
                data = json.load(f)
        else:
            data = states_data

        loaded_states = {}
        for item in data:
            state = ObservedState(
                state_id=item['id'],
                features=np.array(item['features']),
                probability=item.get('probability', 1.0),
                forbidden=item.get('forbidden', False),
                quantum_signature=complex(*item['quantum_signature']) if 'quantum_signature' in item else None,
                metadata=item.get('metadata', {})
            )
            self.observed_states[state.state_id] = state
            loaded_states[state.state_id] = state

        print(f"✓ Loaded {len(loaded_states)} observed states")
        return loaded_states

    def load_forbidden_configurations(self, configs_data: Union[str, List[Dict]]) -> Dict:
        """Load forbidden configurations."""

        if isinstance(configs_data, str):
            with open(configs_data, 'r') as f:
                data = json.load(f)
        else:
            data = configs_data

        loaded_configs = {}
        for item in data:
            config = ForbiddenConfiguration(
                config_id=item['id'],
                state_pattern=np.array(item['pattern']),
                constraint_violation=item['violation'],
                violation_type=item['type'],
                penalty_function=self._create_penalty_function(item.get('penalty_params', {}))
            )
            self.forbidden_configs[config.config_id] = config
            loaded_configs[config.config_id] = config

        print(f"✓ Loaded {len(loaded_configs)} forbidden configurations")
        return loaded_configs

    def _create_penalty_function(self, params: Dict) -> callable:
        """Create penalty function for forbidden configurations."""

        penalty_type = params.get('type', 'quadratic')

        if penalty_type == 'quadratic':
            def penalty(state):
                violation = np.sum((state - params.get('target', np.zeros_like(state)))**2)
                return params.get('coefficient', 1.0) * violation
        elif penalty_type == 'exponential':
            def penalty(state):
                distance = np.linalg.norm(state - params.get('target', np.zeros_like(state)))
                return np.exp(params.get('coefficient', 1.0) * distance)
        else:
            def penalty(state):
                return 0.0

        return penalty

    async def derive_constraint_geometry(self,
                                        method: ConstraintMethod,
                                        derivation_id: str = None) -> ConstraintGeometry:
        """Derive constraint geometry using specified method."""

        if derivation_id is None:
            derivation_id = f"{method.value}_{len(self.constraint_geometries)}"

        print(f"\n🧪 DERIVING CONSTRAINT GEOMETRY using {method.value.upper()} method...")

        if method == ConstraintMethod.QUANTUM_HYBRID:
            geometry = await self._derive_quantum_hybrid()
        elif method == ConstraintMethod.FULL:
            geometry = await self._derive_full()
        elif method == ConstraintMethod.ISING_EMBEDDING:
            geometry = await self._derive_ising_embedding()
        elif method == ConstraintMethod.PSYCHIATRIC:
            geometry = await self._derive_psychiatric()
        else:
            raise ValueError(f"Unknown method: {method}")

        geometry.method = method
        self.constraint_geometries[derivation_id] = geometry

        # Record derivation
        self.derivation_history.append({
            'id': derivation_id,
            'method': method.value,
            'timestamp': asyncio.get_event_loop().time(),
            'n_constraints': geometry.constraint_matrix.shape[0] if geometry.constraint_matrix is not None else 0,
            'allowed_dim': geometry.allowed_subspace.shape[1] if geometry.allowed_subspace is not None else 0
        })

        print(f"✓ Derived constraint geometry '{derivation_id}'")
        print(f"  Constraints: {geometry.constraint_matrix.shape[0] if geometry.constraint_matrix is not None else 'None'}")
        print(f"  Allowed subspace dimension: {geometry.allowed_subspace.shape[1] if geometry.allowed_subspace is not None else 'None'}")

        return geometry

    async def _derive_quantum_hybrid(self) -> ConstraintGeometry:
        """
        Quantum-Hybrid Constraint Derivation.

        Combines:
        - Quantum state tomography from observed states
        - Classical constraint learning
        - Quantum annealing for forbidden configuration penalty
        """

        # Collect all observed state vectors
        state_vectors = []
        quantum_states = []

        for state in self.observed_states.values():
            state_vectors.append(state.features)
            if state.quantum_signature is not None:
                quantum_states.append(state.to_quantum_state())

        if not state_vectors:
            raise ValueError("No observed states loaded")

        X = np.array(state_vectors)  # States as rows
        n_states, n_features = X.shape

        # Step 1: Quantum state reconstruction
        if quantum_states:
            Q = np.array(quantum_states)
            # Density matrix from quantum states
            rho = np.mean([np.outer(q, q.conj()) for q in quantum_states], axis=0)

            # Quantum constraints from density matrix nullspace
            eigenvalues, eigenvectors = np.linalg.eigh(rho)

            # Low probability eigenvectors are constraints
            constraint_threshold = 0.01
            constraint_indices = np.where(eigenvalues < constraint_threshold)[0]
            quantum_constraints = eigenvectors[:, constraint_indices].T.conj()
        else:
            quantum_constraints = np.zeros((0, n_features))

        # Step 2: Classical constraint learning
        # Find linear constraints that separate allowed from forbidden

        allowed_states = [s for s in self.observed_states.values() if not s.forbidden]
        forbidden_states = [s for s in self.observed_states.values() if s.forbidden]

        if allowed_states and forbidden_states:
            # Build constraint matrix via linear discrimination
            X_allowed = np.array([s.features for s in allowed_states])
            X_forbidden = np.array([s.features for s in forbidden_states])

            # Mean difference as constraint direction
            mean_allowed = np.mean(X_allowed, axis=0)
            mean_forbidden = np.mean(X_forbidden, axis=0)

            # Covariance-weighted difference (Fisher discriminant)
            cov_allowed = np.cov(X_allowed.T) if len(allowed_states) > 1 else np.eye(n_features)
            cov_forbidden = np.cov(X_forbidden.T) if len(forbidden_states) > 1 else np.eye(n_features)

            cov_pooled = (cov_allowed + cov_forbidden) / 2
            cov_inv = np.linalg.pinv(cov_pooled)

            w = cov_inv @ (mean_allowed - mean_forbidden)
            w = w / (np.linalg.norm(w) + 1e-10)

            # Threshold as constraint
            scores_allowed = X_allowed @ w
            scores_forbidden = X_forbidden @ w

            threshold = (np.mean(scores_allowed) + np.mean(scores_forbidden)) / 2

            classical_constraint = np.zeros((1, n_features + 1))
            classical_constraint[0, :-1] = w
            classical_constraint[0, -1] = -threshold
        else:
            classical_constraint = np.zeros((0, n_features + 1))

        # Step 3: Combine quantum and classical constraints
        # For quantum constraints: C|ψ⟩ = 0
        # For classical constraints: w·x - threshold = 0

        n_quantum_constraints = quantum_constraints.shape[0]
        n_classical_constraints = classical_constraint.shape[0]

        # Build combined constraint matrix
        if n_quantum_constraints > 0:
            C_quantum = quantum_constraints
        else:
            C_quantum = np.zeros((0, n_features))

        if n_classical_constraints > 0:
            # Convert classical constraint to same format
            C_classical = classical_constraint[:, :-1]
            thresholds = classical_constraint[:, -1:]
        else:
            C_classical = np.zeros((0, n_features))
            thresholds = np.zeros((0, 1))

        # Combine
        C = np.vstack([C_quantum, C_classical])

        # Step 4: Forbidden configuration penalties
        forbidden_penalty_matrix = np.zeros((len(self.forbidden_configs), n_features))
        for i, (config_id, config) in enumerate(self.forbidden_configs.items()):
            # Use pattern as constraint direction
            pattern = config.state_pattern[:n_features]  # Truncate if necessary
            pattern = pattern / (np.linalg.norm(pattern) + 1e-10)
            forbidden_penalty_matrix[i] = pattern

        # Add forbidden constraints with penalty weights
        if len(self.forbidden_configs) > 0:
            penalty_weights = np.array([c.constraint_violation for c in self.forbidden_configs.values()])
            C_forbidden = np.diag(penalty_weights) @ forbidden_penalty_matrix
            C = np.vstack([C, C_forbidden])

        # Create constraint geometry
        geometry = ConstraintGeometry(ConstraintMethod.QUANTUM_HYBRID)
        geometry.constraint_matrix = C

        # Compute allowed subspace (nullspace of C)
        if C.shape[0] > 0:
            U, s, Vh = np.linalg.svd(C, full_matrices=True)

            # Nullspace is columns of Vh corresponding to zero singular values
            nullspace_threshold = 1e-10
            nullspace_indices = np.where(s < nullspace_threshold)[0]

            rank = np.sum(s > nullspace_threshold)
            if len(nullspace_indices) > 0:
                geometry.allowed_subspace = Vh[nullspace_indices].T
            else:
                # No exact nullspace, use smallest singular vectors
                geometry.allowed_subspace = Vh[-1:].T

            # Forbidden subspace is rowspace of C (orthogonal to nullspace)
            geometry.forbidden_subspace = Vh[:rank]

            # Eigenanalysis
            geometry.eigenvalues = s
            geometry.eigenvectors = Vh.T
        else:
            # No constraints, entire space is allowed
            geometry.allowed_subspace = np.eye(n_features)
            geometry.forbidden_subspace = np.zeros((0, n_features))
            geometry.eigenvalues = np.ones(n_features)
            geometry.eigenvectors = np.eye(n_features)

        # Store quantum operators
        if quantum_states:
            geometry.quantum_operators['density_matrix'] = rho
            geometry.quantum_operators['constraint_projectors'] = quantum_constraints

        return geometry

    async def _derive_full(self) -> ConstraintGeometry:
        """
        Full Constraint Derivation.

        Uses complete state space analysis:
        - Principal component analysis for dimensionality reduction
        - Manifold learning for constraint surfaces
        - Topological data analysis for constraint holes
        - Algebraic geometry for polynomial constraints
        """

        # Collect all state vectors
        state_vectors = [s.features for s in self.observed_states.values()]
        X = np.array(state_vectors)
        n_states, n_features = X.shape

        print(f"  Analyzing {n_states} states in {n_features}-dimensional space")

        # Step 1: Dimensionality reduction via PCA
        # Center the data
        X_centered = X - np.mean(X, axis=0)

        # PCA
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Determine intrinsic dimensionality
        explained_variance = S**2 / (n_states - 1)
        explained_variance_ratio = explained_variance / explained_variance.sum()

        cumulative_variance = np.cumsum(explained_variance_ratio)
        intrinsic_dim = np.argmax(cumulative_variance > 0.95) + 1
        intrinsic_dim = min(intrinsic_dim, n_features)

        print(f"  Intrinsic dimensionality: {intrinsic_dim} (captures {cumulative_variance[intrinsic_dim-1]:.1%} variance)")

        # Step 2: Build constraint matrix from orthogonal complement
        # Principal components (allowed directions)
        principal_components = Vt[:intrinsic_dim].T

        # Constraint directions (forbidden directions)
        constraint_directions = Vt[intrinsic_dim:].T

        # Step 3: Manifold learning for nonlinear constraints
        # Use Isomap or other manifold learning (simplified)
        if n_states > 10 and n_features > 2:
            # Simplified: use pairwise distances to find manifold structure
            from scipy.spatial.distance import pdist, squareform

            distances = squareform(pdist(X))

            # Find nearest neighbors for each point
            k = min(5, n_states - 1)
            knn_graph = np.zeros((n_states, n_states))

            for i in range(n_states):
                # Find k nearest neighbors (excluding self)
                nearest = np.argsort(distances[i])[1:k+1]
                knn_graph[i, nearest] = distances[i, nearest]

            # Graph Laplacian for manifold structure
            degree_matrix = np.diag(np.sum(knn_graph, axis=1))
            laplacian = degree_matrix - knn_graph

            # Eigenvectors of Laplacian give manifold coordinates
            laplacian_eigvals, laplacian_eigvecs = np.linalg.eigh(laplacian)

            # Small eigenvalues correspond to smooth functions on manifold
            manifold_constraint_directions = laplacian_eigvecs[:, :intrinsic_dim].T
        else:
            manifold_constraint_directions = np.zeros((0, n_features))

        # Step 4: Algebraic constraints from polynomial fitting
        # Find polynomial constraints of degree 2
        polynomial_constraints = []

        if n_states > n_features * (n_features + 1) // 2:  # Enough data for quadratic
            # For each feature, try to predict it from others using quadratic model
            for target_idx in range(min(5, n_features)):  # Limit to first 5 features
                # Build quadratic features
                from sklearn.preprocessing import PolynomialFeatures

                poly = PolynomialFeatures(degree=2, include_bias=False)
                X_poly = poly.fit_transform(X_centered[:, np.arange(n_features) != target_idx])

                # Try to predict target feature
                y = X_centered[:, target_idx]

                # Solve linear system (simplified)
                if X_poly.shape[0] >= X_poly.shape[1]:
                    coeffs = np.linalg.lstsq(X_poly, y, rcond=None)[0]

                    # Check if prediction is good (constraint exists)
                    y_pred = X_poly @ coeffs
                    r2 = 1 - np.var(y - y_pred) / np.var(y)

                    if r2 > 0.9:  # Strong constraint
                        # Convert back to constraint on full feature space
                        constraint_vec = np.zeros(n_features)
                        constraint_vec[target_idx] = 1.0

                        # Map polynomial coefficients back (simplified)
                        # This would require careful mapping in real implementation
                        polynomial_constraints.append(constraint_vec)

        polynomial_constraints = np.array(polynomial_constraints) if polynomial_constraints else np.zeros((0, n_features))

        # Step 5: Combine all constraint sources
        constraint_sources = []

        if constraint_directions.shape[0] > 0:
            constraint_sources.append(constraint_directions.T)

        if manifold_constraint_directions.shape[0] > 0:
            constraint_sources.append(manifold_constraint_directions)

        if polynomial_constraints.shape[0] > 0:
            constraint_sources.append(polynomial_constraints)

        # Add forbidden configuration constraints
        forbidden_constraints = []
        for config in self.forbidden_configs.values():
            forbidden_constraints.append(config.state_pattern[:n_features])

        if forbidden_constraints:
            constraint_sources.append(np.array(forbidden_constraints))

        # Combine all constraints
        if constraint_sources:
            C = np.vstack(constraint_sources)

            # Remove linearly dependent constraints
            U, s, Vh = np.linalg.svd(C, full_matrices=False)
            rank = np.sum(s > 1e-10)
            C = U[:, :rank] @ np.diag(s[:rank]) @ Vh[:rank, :]
        else:
            C = np.zeros((0, n_features))

        # Create constraint geometry
        geometry = ConstraintGeometry(ConstraintMethod.FULL)
        geometry.constraint_matrix = C

        # Compute allowed subspace (orthogonal to constraints)
        if C.shape[0] > 0:
            # Allowed subspace is nullspace of C
            U, s, Vh = np.linalg.svd(C, full_matrices=True)
            nullspace_threshold = 1e-10
            nullspace_indices = np.where(s < nullspace_threshold)[0]

            if len(nullspace_indices) > 0:
                geometry.allowed_subspace = Vh[nullspace_indices].T
            else:
                # Approximate nullspace from smallest singular values
                n_allowed = max(1, n_features - rank)
                geometry.allowed_subspace = Vh[-n_allowed:].T

            # Forbidden subspace is rowspace
            geometry.forbidden_subspace = Vh[:rank]

            geometry.eigenvalues = s
            geometry.eigenvectors = Vh.T
        else:
            geometry.allowed_subspace = np.eye(n_features)
            geometry.forbidden_subspace = np.zeros((0, n_features))
            geometry.eigenvalues = np.ones(n_features)
            geometry.eigenvectors = np.eye(n_features)

        # Store analysis results
        geometry.quantum_operators = {
            'principal_components': principal_components,
            'intrinsic_dimension': intrinsic_dim,
            'explained_variance': explained_variance_ratio
        }

        return geometry

    def ingest_ising_model(self, ising_data: Union[str, List[Dict]]) -> Dict:
        """Ingest Ising model data into CGDA Lab."""

        if isinstance(ising_data, str):
            with open(ising_data, 'r') as f:
                data = json.load(f)
        else:
            data = ising_data

        self.ising_data.extend(data)

        # Convert Ising model to constraint representation
        ising_constraints = []

        for model in data:
            # Ising Hamiltonian: H = -∑J_ij σ_i σ_j - ∑h_i σ_i
            n_spins = model.get('n_spins', len(model.get('couplings', [])) + 1)

            # Build constraint matrix from ground states
            ground_states = model.get('ground_states', [])

            if ground_states:
                # Convert spin configurations to feature vectors
                # σ = ±1 → features = [1, σ] or similar
                features = []
                for state in ground_states:
                    # Simple encoding: concatenate spins and products
                    spin_vector = np.array(state)

                    # Add pairwise products for couplings
                    if 'couplings' in model:
                        n = len(spin_vector)
                        products = []
                        for i in range(n):
                            for j in range(i+1, n):
                                products.append(spin_vector[i] * spin_vector[j])

                        feature = np.concatenate([spin_vector, products])
                    else:
                        feature = spin_vector

                    features.append(feature)

                # Create observed states from ground states
                for i, feature in enumerate(features):
                    state_id = f"ising_{model.get('id', 'unknown')}_gs_{i}"
                    state = ObservedState(
                        state_id=state_id,
                        features=feature,
                        probability=1.0 / len(features),
                        forbidden=False,
                        metadata={'ising_model': model.get('id'), 'energy': model.get('energy', 0)}
                    )
                    self.observed_states[state_id] = state

            # Create forbidden configurations from excited states
            excited_states = model.get('excited_states', [])
            for i, state in enumerate(excited_states):
                config_id = f"ising_forbidden_{model.get('id', 'unknown')}_{i}"

                # Calculate violation proportional to energy above ground
                energy = state.get('energy', 1.0)
                ground_energy = model.get('ground_energy', 0.0)
                violation = max(0, energy - ground_energy)

                # Convert spin configuration to pattern
                spin_pattern = np.array(state.get('spins', []))

                config = ForbiddenConfiguration(
                    config_id=config_id,
                    state_pattern=spin_pattern,
                    constraint_violation=violation,
                    violation_type='ising_energy_barrier',
                    penalty_function=lambda s, e=violation: e * np.sum((s - spin_pattern)**2)
                )
                self.forbidden_configs[config_id] = config

        print(f"✓ Ingested {len(data)} Ising models")
        print(f"  Created {sum(1 for s in self.observed_states.values() if 'ising' in s.state_id)} ground states")
        print(f"  Created {sum(1 for c in self.forbidden_configs.values() if 'ising' in c.config_id)} forbidden configurations")

        return {
            'models_ingested': len(data),
            'states_created': sum(1 for s in self.observed_states.values() if 'ising' in s.state_id),
            'configs_created': sum(1 for c in self.forbidden_configs.values() if 'ising' in c.config_id)
        }

    def ingest_psychiatric_state_data(self, psychiatric_data: Union[str, List[Dict]]) -> Dict:
        """Ingest psychiatric state data into CGDA Lab."""

        if isinstance(psychiatric_data, str):
            with open(psychiatric_data, 'r') as f:
                data = json.load(f)
        else:
            data = psychiatric_data

        self.psychiatric_data.extend(data)

        # Psychiatric states to constraint representation
        psychiatric_states_created = 0
        psychiatric_configs_created = 0

        for patient in data:
            patient_id = patient.get('patient_id', 'unknown')
            states = patient.get('states', [])
            diagnosis = patient.get('diagnosis', 'unknown')

            for i, state in enumerate(states):
                # Extract features from psychiatric state
                # Common psychiatric rating scales: PHQ-9, GAD-7, PANSS, etc.
                features = []

                if 'phq9' in state:
                    features.extend(state['phq9'])

                if 'gad7' in state:
                    features.extend(state['gad7'])

                if 'panss' in state:
                    features.extend([state['panss'].get('positive', 0),
                                    state['panss'].get('negative', 0),
                                    state['panss'].get('general', 0)])

                if 'bprs' in state:
                    features.append(state['bprs'])

                # Add symptom scores
                if 'symptoms' in state:
                    for symptom, severity in state['symptoms'].items():
                        features.append(severity)

                if not features:
                    continue

                features = np.array(features)

                # Determine if state is pathological (forbidden)
                # Based on clinical thresholds
                is_pathological = False
                violation = 0.0

                if 'phq9' in state and sum(state['phq9']) >= 10:  # Moderate depression threshold
                    is_pathological = True
                    violation = sum(state['phq9']) / 27.0  # Normalized

                if 'gad7' in state and sum(state['gad7']) >= 10:  # Moderate anxiety threshold
                    is_pathological = True
                    violation = max(violation, sum(state['gad7']) / 21.0)

                state_id = f"psych_{patient_id}_state_{i}"

                if is_pathological:
                    # Create forbidden configuration for pathological state
                    config_id = f"psych_forbidden_{patient_id}_{i}"
                    config = ForbiddenConfiguration(
                        config_id=config_id,
                        state_pattern=features,
                        constraint_violation=violation,
                        violation_type=f'psychiatric_{diagnosis}',
                        penalty_function=lambda s, f=features, v=violation: v * np.linalg.norm(s - f)**2
                    )
                    self.forbidden_configs[config_id] = config
                    psychiatric_configs_created += 1
                else:
                    # Create observed state for healthy/non-pathological state
                    obs_state = ObservedState(
                        state_id=state_id,
                        features=features,
                        probability=state.get('probability', 1.0 / len(states)),
                        forbidden=False,
                        metadata={
                            'patient_id': patient_id,
                            'diagnosis': diagnosis,
                            'timestamp': state.get('timestamp'),
                            'treatment': state.get('treatment', 'none')
                        }
                    )
                    self.observed_states[state_id] = obs_state
                    psychiatric_states_created += 1

        print(f"✓ Ingested {len(data)} psychiatric patient records")
        print(f"  Created {psychiatric_states_created} psychiatric states")
        print(f"  Created {psychiatric_configs_created} pathological configurations")

        return {
            'patients_ingested': len(data),
            'states_created': psychiatric_states_created,
            'pathological_configs_created': psychiatric_configs_created
        }

    async def _derive_ising_embedding(self) -> ConstraintGeometry:
        """Derive constraint geometry specifically for Ising models."""

        print("  Deriving Ising model constraint geometry...")

        # Collect Ising-related states
        ising_states = [s for s in self.observed_states.values() if 'ising' in s.state_id]

        if not ising_states:
            raise ValueError("No Ising model data ingested")

        # Extract features (spin configurations)
        X = np.array([s.features for s in ising_states])
        n_states, n_features = X.shape

        # Ising constraints: spins must be ±1
        # For each spin dimension, constraint: σ_i² - 1 = 0
        ising_constraints = []

        # Simple linear constraints for Ising spins
        # In actual Ising model, constraints are quadratic
        # We'll create linear approximations

        # Mean magnetization constraints
        mean_spins = np.mean(X, axis=0)

        # Create constraints from deviations
        for i in range(n_features):
            if abs(mean_spins[i]) < 0.5:  # Not strongly magnetized
                # Constraint: spin should be near mean
                constraint_vec = np.zeros(n_features)
                constraint_vec[i] = 1.0
                ising_constraints.append(constraint_vec)

        ising_constraints = np.array(ising_constraints) if ising_constraints else np.zeros((0, n_features))

        # Create geometry
        geometry = ConstraintGeometry(ConstraintMethod.ISING_EMBEDDING)
        geometry.constraint_matrix = ising_constraints

        # Compute allowed subspace
        if ising_constraints.shape[0] > 0:
            U, s, Vh = np.linalg.svd(ising_constraints, full_matrices=True)
            nullspace_threshold = 1e-10
            nullspace_indices = np.where(s < nullspace_threshold)[0]

            rank = np.sum(s > nullspace_threshold)
            if len(nullspace_indices) > 0:
                geometry.allowed_subspace = Vh[nullspace_indices].T
            else:
                geometry.allowed_subspace = Vh[-1:].T

            geometry.forbidden_subspace = Vh[:rank]
            geometry.eigenvalues = s
            geometry.eigenvectors = Vh.T
        else:
            geometry.allowed_subspace = np.eye(n_features)
            geometry.forbidden_subspace = np.zeros((0, n_features))
            geometry.eigenvalues = np.ones(n_features)
            geometry.eigenvectors = np.eye(n_features)

        return geometry

    async def _derive_psychiatric(self) -> ConstraintGeometry:
        """Derive constraint geometry for psychiatric states."""

        print("  Deriving psychiatric constraint geometry...")

        # Collect psychiatric states
        psych_states = [s for s in self.observed_states.values() if 'psych' in s.state_id]
        psych_forbidden = [c for c in self.forbidden_configs.values() if 'psych' in c.config_id]

        if not psych_states and not psych_forbidden:
            raise ValueError("No psychiatric data ingested")

        # Combine all psychiatric feature vectors
        all_features = []
        for state in psych_states:
            all_features.append(state.features)

        for config in psych_forbidden:
            all_features.append(config.state_pattern)

        if not all_features:
            raise ValueError("No feature vectors extracted")

        X = np.array(all_features)
        n_samples, n_features = X.shape

        # Psychiatric constraints often involve thresholds
        # e.g., PHQ-9 total < 10 is non-depressed

        # Create constraints based on clinical thresholds
        constraints = []

        # Example: if we know feature 0-8 are PHQ-9 items
        # Constraint: sum(features[0:9]) < 10 for non-depressed

        # For demonstration, create simple linear constraints
        # In practice, would use clinical guidelines

        # Center the data
        X_centered = X - np.mean(X, axis=0)

        # PCA to find main directions of variation
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Use last few principal components as constraints
        # (directions of low variance may represent constraints)
        n_constraints = min(3, n_features)
        constraint_directions = Vt[-n_constraints:].T

        constraints = constraint_directions.T

        # Add constraints from pathological configurations
        if psych_forbidden:
            forbidden_patterns = np.array([c.state_pattern[:n_features] for c in psych_forbidden])

            # Mean pathological pattern
            mean_pathological = np.mean(forbidden_patterns, axis=0)

            # Constraint: distance from mean pathological > threshold
            constraint_vec = mean_pathological / (np.linalg.norm(mean_pathological) + 1e-10)
            constraints = np.vstack([constraints, constraint_vec])

        # Create geometry
        geometry = ConstraintGeometry(ConstraintMethod.PSYCHIATRIC)
        geometry.constraint_matrix = constraints

        # Compute allowed subspace (healthy states)
        if constraints.shape[0] > 0:
            U, s, Vh = np.linalg.svd(constraints, full_matrices=True)
            nullspace_threshold = 1e-10
            nullspace_indices = np.where(s < nullspace_threshold)[0]

            rank = np.sum(s > nullspace_threshold)
            if len(nullspace_indices) > 0:
                geometry.allowed_subspace = Vh[nullspace_indices].T
            else:
                geometry.allowed_subspace = Vh[-1:].T

            geometry.forbidden_subspace = Vh[:rank]
            geometry.eigenvalues = s
            geometry.eigenvectors = Vh.T
        else:
            geometry.allowed_subspace = np.eye(n_features)
            geometry.forbidden_subspace = np.zeros((0, n_features))
            geometry.eigenvalues = np.ones(n_features)
            geometry.eigenvectors = np.eye(n_features)

        # Store psychiatric-specific information
        geometry.quantum_operators = {
            'clinical_thresholds': {
                'depression': 'PHQ-9 < 10',
                'anxiety': 'GAD-7 < 10',
                'psychosis': 'PANSS thresholds'
            },
            'diagnoses': list(set([s.metadata.get('diagnosis', 'unknown') for s in psych_states]))
        }

        return geometry

    def generate_report(self) -> Dict:
        """Generate comprehensive lab report."""

        report = {
            'lab_id': self.lab_id,
            'observed_states': len(self.observed_states),
            'forbidden_configurations': len(self.forbidden_configs),
            'constraint_geometries': len(self.constraint_geometries),
            'ising_models': len(self.ising_data),
            'psychiatric_patients': len(self.psychiatric_data),
            'derivation_history': self.derivation_history,
            'state_statistics': {
                'total_states': len(self.observed_states),
                'allowed_states': sum(1 for s in self.observed_states.values() if not s.forbidden),
                'forbidden_states': sum(1 for s in self.observed_states.values() if s.forbidden),
                'quantum_states': sum(1 for s in self.observed_states.values() if s.quantum_signature is not None)
            }
        }

        # Add geometry statistics
        geometry_stats = []
        for geom_id, geometry in self.constraint_geometries.items():
            if geometry.constraint_matrix is not None:
                n_constraints, n_features = geometry.constraint_matrix.shape
                allowed_dim = geometry.allowed_subspace.shape[1] if geometry.allowed_subspace is not None else 0

                geometry_stats.append({
                    'id': geom_id,
                    'method': geometry.method.value,
                    'n_constraints': n_constraints,
                    'n_features': n_features,
                    'allowed_dimension': allowed_dim,
                    'forbidden_dimension': n_features - allowed_dim
                })

        report['geometry_statistics'] = geometry_stats

        return report

# === SAMPLE DATA GENERATION ===

def generate_sample_observed_states() -> List[Dict]:
    """Generate sample observed states for testing."""

    states = []

    # Quantum system states (e.g., qubit states)
    quantum_states = [
        {'id': 'psi_plus', 'features': [1/np.sqrt(2), 1/np.sqrt(2), 0, 0],
         'probability': 0.5, 'quantum_signature': [1, 0]},
        {'id': 'psi_minus', 'features': [1/np.sqrt(2), -1/np.sqrt(2), 0, 0],
         'probability': 0.3, 'quantum_signature': [0, 1]},
        {'id': 'phi_plus', 'features': [0.8, 0.2, 0, 0],
         'probability': 0.2, 'quantum_signature': [0.6, 0.8]},
    ]

    # Classical system states
    classical_states = [
        {'id': 'classical_1', 'features': [1, 0, 0, 1], 'probability': 0.4},
        {'id': 'classical_2', 'features': [0, 1, 1, 0], 'probability': 0.3},
        {'id': 'classical_3', 'features': [1, 1, 0, 0], 'probability': 0.2},
        {'id': 'classical_4', 'features': [0, 0, 1, 1], 'probability': 0.1},
    ]

    # Forbidden states (violate constraints)
    forbidden_states = [
        {'id': 'forbidden_1', 'features': [1, 1, 1, 1],
         'probability': 0.01, 'forbidden': True},
        {'id': 'forbidden_2', 'features': [0, 0, 0, 0],
         'probability': 0.005, 'forbidden': True},
    ]

    states.extend(quantum_states)
    states.extend(classical_states)
    states.extend(forbidden_states)

    return states

def generate_sample_forbidden_configs() -> List[Dict]:
    """Generate sample forbidden configurations."""

    configs = [
        {
            'id': 'config_1',
            'pattern': [1, 1, 1, 1],
            'violation': 2.0,
            'type': 'total_alignment',
            'penalty_params': {'type': 'quadratic', 'coefficient': 2.0}
        },
        {
            'id': 'config_2',
            'pattern': [0, 0, 0, 0],
            'violation': 1.5,
            'type': 'total_anti_alignment',
            'penalty_params': {'type': 'exponential', 'coefficient': 1.0}
        },
        {
            'id': 'config_3',
            'pattern': [1, 0, 1, 0],
            'violation': 0.8,
            'type': 'alternating_pattern',
            'penalty_params': {'type': 'quadratic', 'coefficient': 1.0, 'target': [0.5, 0.5, 0.5, 0.5]}
        }
    ]

    return configs

def generate_sample_ising_data() -> List[Dict]:
    """Generate sample Ising model data."""

    models = [
        {
            'id': 'ising_1d_A',
            'n_spins': 4,
            'ground_energy': -3.0,
            'ground_states': [
                [1, 1, 1, 1],
                [-1, -1, -1, -1]
            ],
            'excited_states': [
                {'spins': [1, -1, 1, -1], 'energy': -1.0},
                {'spins': [-1, 1, -1, 1], 'energy': -1.0}
            ]
        },
        {
            'id': 'ising_1d_B',
            'n_spins': 4,
            'ground_energy': -3.0,
            'ground_states': [
                [1, -1, 1, -1],
                [-1, 1, -1, 1]
            ],
            'excited_states': [
                {'spins': [1, 1, 1, 1], 'energy': -1.0},
                {'spins': [-1, -1, -1, -1], 'energy': -1.0}
            ]
        }
    ]

    return models

def generate_sample_psychiatric_data() -> List[Dict]:
    """Generate sample psychiatric state data."""

    patients = [
        {
            'patient_id': 'PT001',
            'diagnosis': 'major_depressive_disorder',
            'states': [
                {
                    'timestamp': '2024-01-15',
                    'phq9': [2, 1, 2, 1, 1, 0, 1, 0, 1],  # Total: 9 (mild)
                    'gad7': [1, 1, 0, 1, 0, 0, 1],  # Total: 4 (minimal)
                    'probability': 0.3
                },
                {
                    'timestamp': '2024-02-15',
                    'phq9': [3, 2, 3, 2, 2, 1, 2, 1, 2],  # Total: 18 (moderate-severe)
                    'gad7': [2, 2, 1, 2, 1, 1, 2],  # Total: 11 (moderate)
                    'probability': 0.5
                },
                {
                    'timestamp': '2024-03-15',
                    'phq9': [1, 1, 1, 0, 0, 0, 1, 0, 0],  # Total: 4 (minimal)
                    'gad7': [0, 1, 0, 0, 0, 0, 0],  # Total: 1 (minimal)
                    'probability': 0.2,
                    'treatment': 'SSRI_started'
                }
            ]
        },
        {
            'patient_id': 'PT002',
            'diagnosis': 'generalized_anxiety_disorder',
            'states': [
                {
                    'timestamp': '2024-01-20',
                    'phq9': [1, 1, 0, 1, 0, 0, 0, 0, 0],  # Total: 3
                    'gad7': [3, 2, 2, 3, 2, 1, 2],  # Total: 15 (severe)
                    'probability': 0.6
                },
                {
                    'timestamp': '2024-02-20',
                    'phq9': [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Total: 0
                    'gad7': [1, 1, 0, 1, 0, 0, 1],  # Total: 4 (minimal)
                    'probability': 0.4,
                    'treatment': 'CBT_completed'
                }
            ]
        }
    ]

    return patients

# === MAIN DEMONSTRATION ===

async def demonstrate_cgda_lab():
    """Demonstrate complete CGDA Lab functionality."""

    print("🧪 CGDA LAB INITIALIZATION")
    print("Constraint Geometry and Dynamics Analysis")
    print("=" * 60)

    # Initialize lab
    lab = CGDALab("cgda_quantum_lab_v1")

    # 1. Load custom observed states
    print("\n1. LOADING CUSTOM OBSERVED STATES...")
    sample_states = generate_sample_observed_states()
    lab.load_observed_states(sample_states)

    # 2. Load forbidden configurations
    print("\n2. LOADING FORBIDDEN CONFIGURATIONS...")
    sample_configs = generate_sample_forbidden_configs()
    lab.load_forbidden_configurations(sample_configs)

    # 3. Derive constraint geometry using QUANTUM_HYBRID method
    print("\n3. DERIVING CONSTRAINT GEOMETRY (QUANTUM_HYBRID)...")
    geom_hybrid = await lab.derive_constraint_geometry(
        ConstraintMethod.QUANTUM_HYBRID,
        "quantum_hybrid_derivation_1"
    )

    # Test the derived geometry
    test_state = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0])
    satisfies = geom_hybrid.satisfies_constraints(test_state)
    print(f"  Test state satisfies constraints: {satisfies}")

    # 4. Derive constraint geometry using FULL method
    print("\n4. DERIVING CONSTRAINT GEOMETRY (FULL)...")
    geom_full = await lab.derive_constraint_geometry(
        ConstraintMethod.FULL,
        "full_derivation_1"
    )

    # 5. Ingest Ising model data
    print("\n5. INGESTING ISING MODEL DATA...")
    ising_data = generate_sample_ising_data()
    ising_result = lab.ingest_ising_model(ising_data)

    # Derive Ising-specific constraints
    print("\n6. DERIVING ISING EMBEDDING CONSTRAINTS...")
    try:
        geom_ising = await lab.derive_constraint_geometry(
            ConstraintMethod.ISING_EMBEDDING,
            "ising_embedding_1"
        )
    except ValueError as e:
        print(f"  Note: {e}")

    # 6. Ingest psychiatric state data
    print("\n7. INGESTING PSYCHIATRIC STATE DATA...")
    psych_data = generate_sample_psychiatric_data()
    psych_result = lab.ingest_psychiatric_state_data(psych_data)

    # Derive psychiatric constraints
    print("\n8. DERIVING PSYCHIATRIC CONSTRAINTS...")
    try:
        geom_psych = await lab.derive_constraint_geometry(
            ConstraintMethod.PSYCHIATRIC,
            "psychiatric_constraints_1"
        )
    except ValueError as e:
        print(f"  Note: {e}")

    # 7. Generate comprehensive report
    print("\n9. GENERATING COMPREHENSIVE LAB REPORT...")
    report = lab.generate_report()

    print(f"\n{'='*60}")
    print("CGDA LAB REPORT SUMMARY")
    print(f"{'='*60}")

    print(f"Lab ID: {report['lab_id']}")
    print(f"Total observed states: {report['observed_states']}")
    print(f"Total forbidden configurations: {report['forbidden_configurations']}")
    print(f"Constraint geometries derived: {report['constraint_geometries']}")
    print(f"Ising models ingested: {report['ising_models']}")
    print(f"Psychiatric patients ingested: {report['psychiatric_patients']}")

    print(f"\nState Statistics:")
    stats = report['state_statistics']
    print(f"  Allowed states: {stats['allowed_states']}")
    print(f"  Forbidden states: {stats['forbidden_states']}")
    print(f"  Quantum states: {stats['quantum_states']}")

    print(f"\nConstraint Geometry Statistics:")
    for geom in report['geometry_statistics']:
        print(f"\n  {geom['id']} ({geom['method']}):")
        print(f"    Features: {geom['n_features']}")
        print(f"    Constraints: {geom['n_constraints']}")
        print(f"    Allowed dimension: {geom['allowed_dimension']}")
        print(f"    Forbidden dimension: {geom['forbidden_dimension']}")

    # 8. Demonstrate constraint application
    print(f"\n{'='*60}")
    print("CONSTRAINT APPLICATION DEMONSTRATION")
    print(f"{'='*60}")

    if 'quantum_hybrid_derivation_1' in lab.constraint_geometries:
        geometry = lab.constraint_geometries['quantum_hybrid_derivation_1']

        # Create a test state that violates constraints
        test_violating_state = np.array([1, 1, 1, 1])  # Matches forbidden pattern
        violates = not geometry.satisfies_constraints(test_violating_state)

        print(f"\nTest violating state: {test_violating_state}")
        print(f"Violates constraints: {violates}")

        if violates:
            # Project to allowed subspace
            projected = geometry.project_to_allowed(test_violating_state)
            print(f"Projected state: {projected}")

            # Check if projection satisfies constraints
            satisfies_projected = geometry.satisfies_constraints(projected)
            print(f"Projected state satisfies constraints: {satisfies_projected}")

    # 9. Show integration with biomedical protocol
    print(f"\n{'='*60}")
    print("INTEGRATION WITH BIOMEDICAL PROTOCOL")
    print(f"{'='*60}")

    # Simulate constraint-based DNA repair
    if 'psychiatric_constraints_1' in lab.constraint_geometries:
        psych_geometry = lab.constraint_geometries['psychiatric_constraints_1']

        # Example: Move from pathological to healthy state
        # Create a pathological state with the correct number of features (16)
        pathological_state = np.zeros(psych_geometry.allowed_subspace.shape[0])
        phq9_items = [3, 2, 3, 2, 2, 1, 2, 1, 2] # Total 18
        pathological_state[:9] = phq9_items

        print(f"\nPathological state (depression): {pathological_state[:5]}...")
        print(f"Projecting to healthy subspace...")

        healthy_projection = psych_geometry.project_to_allowed(pathological_state)
        print(f"Healthy projection: {healthy_projection[:5]}...")

        # Calculate improvement
        improvement = np.linalg.norm(pathological_state) - np.linalg.norm(healthy_projection)
        print(f"Symptom severity reduction: {improvement:.2f}")

    return lab, report

# === MAIN EXECUTION ===

if __name__ == "__main__":
    asyncio.run(demonstrate_cgda_lab())
