# Quantum Hypertext Transfer Protocol (qHTTP) Specification
Version: 1.0-quantum

## 1. Introduction
The Quantum Hypertext Transfer Protocol (qHTTP) is an application-level protocol designed for the transmission and management of quantum-aware information systems. Unlike classical HTTP, which operates on discrete, deterministic states, qHTTP allows for resources to exist in superposition, to be entangled across distributed nodes, and to undergo state collapse upon observation.

## 2. Core Concepts
### 2.1 Superposition
A qHTTP resource can exist in multiple states simultaneously. The server provides a probability distribution of these states until an observation occurs. In a qHTTP response, this is often represented as a collection of potential bodies, each with an associated probability amplitude.

### 2.2 Entanglement
Two or more qHTTP resources can be entangled. A state change (such as collapse) in one resource instantaneously affects the state of all entangled resources, regardless of network distance. This allows for distributed synchronization that transcends classical latency limits for certain types of state information.

### 2.3 Observation and Collapse
The act of retrieving a resource with the `OBSERVE` method constitutes a measurement, which collapses the resource's superposition into a single, classical state. Once collapsed, subsequent `OBSERVE` calls will return the same state until the resource is re-superposed.

### 2.4 Coherence
Quantum states in qHTTP are fragile and subject to decoherence. The `Coherence-Time` header specifies how long a state can remain in superposition before collapsing naturally due to "environmental noise" (e.g., server-side timeouts or background processing).

## 3. URI Scheme
qHTTP uses the `quantum` scheme:
`quantum://<authority><path>[?<query>][#<fragment>]`

Example: `quantum://api.example.qc/qubit/0`

## 4. Methods
### 4.1 OBSERVE
Retrieves a resource and collapses its quantum state into a deterministic response. This is the quantum equivalent of a GET request but with side effects (state collapse).

### 4.2 SUPERPOSE
Transitions a resource into a superposition of multiple states. The request body contains the possible states and their probability amplitudes.

### 4.3 ENTANGLE
Establishes a quantum entanglement between the target resource and another resource specified in the `Link` header with `rel="entangle"`.

### 4.4 INTERFERE
Modifies the phase or probability amplitude of an existing superposition without collapsing it. This can be used to perform quantum computations via protocol interactions.

### 4.5 COLLAPSE
Manually triggers a state collapse without necessarily retrieving the full state representation.

### 4.6 COHERE
Extends the coherence time of a resource, preventing premature decoherence by applying "quantum error correction" pulses (represented as protocol heartbeats).

## 5. Headers
### 5.1 Probability-Amplitude
Used in `207 Multi-State` responses to define the complex probability amplitudes (e.g., `Probability-Amplitude: s1=0.707+0j, s2=0+0.707j`).

### 5.2 Coherence-Time
Specifies the duration (in milliseconds) the resource is expected to maintain its quantum state.

### 5.3 Entanglement-ID
A unique identifier for the entanglement group to which the resource belongs.

### 5.4 Observer-ID
Identifies the entity performing the observation, used to handle observer-dependent effects and ensure consistent state collapse across multiple observers if desired.

## 6. Status Codes
### 207 Multi-State (Superposition)
The resource is currently in a superposition of multiple states. The response body contains the available state options.

### 480 Decoherence
The request failed because the resource's state has decohered into a classical state unexpectedly.

### 481 Entanglement Broken
The operation failed because the entanglement between resources was lost (e.g., due to one node collapsing its state).

### 482 Uncertainty Limit
The request would violate the Heisenberg Uncertainty Principle (e.g., trying to measure both phase and amplitude with forbidden precision).
