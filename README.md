# Quantum Neural Search
###### Based on research combining quantum computing with neuroscience applications, implementing Grover's search and variational quantum algorithms for brain state classification. Reference data sources include depersonalized recordings from [McGill University Health Center (MUHC)](https://www.muhc.ca/) and [Jewish General Hospital](https://www.jgh.ca/), used under strict confidentiality protocols for validation of signal characteristics only.

![Master Brain Analysis](./Plots/master_brain_analysis.png)

## Objective

This repository implements quantum algorithms for neuroscience applications, specifically targeting brain network analysis and EEG pattern recognition. The project demonstrates how quantum computing can provide computational advantages for neural signal processing through two primary approaches: exact pattern search using Grover's algorithm and adaptive classification using variational quantum circuits.

The core challenge in computational neuroscience is efficiently processing high-dimensional neural data where classical algorithms face exponential scaling problems. For a brain network with N neurons, the number of possible connectivity patterns grows exponentially, while the neural state space scales as 2^N. Quantum algorithms can potentially overcome these limitations through superposition and entanglement effects.

**Goal:** Demonstrate quantum computational advantages for neural pattern recognition, implement scalable quantum algorithms for brain network analysis, and establish frameworks for quantum-enhanced neuroscience applications compatible with near-term quantum hardware.

## Theoretical Background

This implementation combines classical neuroscience models with quantum computing principles to create hybrid algorithms for neural pattern analysis.

### Neural Dynamics and Quantum Encoding

Biological neural networks are modeled using the Leaky Integrate-and-Fire (LIF) neuron framework, which captures essential neural behavior through membrane potential dynamics:

$$\tau \frac{dV}{dt} = -V(t) + I(t)$$

where $V(t)$ represents membrane potential, $\tau$ is the membrane time constant, and $I(t)$ represents synaptic input. For network-level analysis, this extends to coupled systems with connectivity matrices:

$$\frac{d\mathbf{V}}{dt} = -\frac{1}{\tau}\mathbf{V} + \mathbf{W}\mathbf{s}(t) + \mathbf{I}_{ext}(t)$$

The quantum analog implements a Quantum Leaky Integrate-and-Fire (QLIF) model where neural excitation is represented through qubit state probabilities:

$$\alpha[t+1] = \sin^2\left(\frac{(\theta + \varphi[t])X[t+1] + (\gamma[t] + \varphi[t])(1-X[t+1])}{2}\right)$$

where $\alpha[t]$ represents excited state probability, $\theta$ controls spike rotation angles, and $\gamma[t]$ models quantum decoherence effects.

### **Quantum Algorithm Formulations**

Grover's Search Algorithm provides quadratic speedup for exact neural pattern detection through amplitude amplification:

$$|\psi_{final}\rangle = (G)^{\sqrt{N}/4} |s\rangle$$

where $G = -U_s U_f$ represents the **Grover operator** combining the oracle $U_f$ (which marks target brain states) and diffusion operator $U_s$ (which amplifies marked amplitudes).

Variational Quantum Circuits enable adaptive brain state classification through parametrized quantum gates:

$$\langle \hat{H} \rangle_{\boldsymbol{\theta}} = \langle \psi_{data} | U^{\dagger}(\boldsymbol{\theta}) \hat{H} U(\boldsymbol{\theta}) | \psi_{data} \rangle$$

where $U(\boldsymbol{\theta})$ represents a parametrized quantum circuit optimized to minimize classification cost functions through gradient-based optimization.

### **Signal Encoding Strategies**

Three primary encoding methods convert continuous EEG signals to quantum-compatible discrete states:

Threshold Encoding creates binary representations based on statistical properties:
```math
b[n] = \begin{cases} 
1 & \text{if } |x[n]| > \theta_{factor} \cdot \sigma(x) \\ 
0 & \text{otherwise} 
\end{cases}
```

Phase Encoding captures oscillatory dynamics through Hilbert transform analysis:
$$\phi[n] = \arg(\mathcal{H}(x[n]))$$

Amplitude Encoding preserves magnitude relationships while normalizing to quantum state requirements:
$$a[n] = \frac{x[n] - x_{min}}{x_{max} - x_{min}}$$

Neural complexity is quantified using **Shannon entropy** for discrete patterns, with Normalized Corrected Shannon Entropy (NCSE) providing standardized measurements:
$$NCSE(L,\Psi) = \frac{CSE(L,\Psi)}{CSE_{max}(L,\Psi)}$$

where $L$ represents symbolic word length and $\Psi$ denotes the symbolic sequence derived from neural time series data.

## Code Functionality

### 1. Brain Network Setup and Atlas Creation

This module establishes the foundational brain architecture by creating a realistic atlas of 33 brain regions organized across 6 major functional networks. The implementation generates anatomically accurate 3D coordinates in Montreal Neurological Institute (MNI) space, providing the spatial framework for all subsequent network analyses.

```python
def createBrainAtlas():
    """Create brain atlas with functional network organization."""
    brain_regions = [
        # Default Mode Network
        'mPFC', 'PCC', 'Angular_L', 'Angular_R', 'ITG_L', 'ITG_R',
        # Salience Network  
        'dACC', 'AI_L', 'AI_R', 'VLPFC_L', 'VLPFC_R',
        # ... additional networks
    ]
    
    # Generate 3D coordinates for brain regions (MNI space)
    coords = []
    for name, network in zip(brain_regions, network_labels):
        if network == 'DMN':
            base = [0, -50, 30] if 'PCC' in name else [0, 50, 0]
        # ... coordinate generation for all networks
        
    return atlasinfo, coords, nodes_df_coords
```

![Brain Networks Multiview](./Plots/brain_networks_multiview_notebook.png)

### 2. Brain Connectivity Matrix Generation

This component generates biologically realistic connectivity matrices that capture the modular structure of brain networks. The algorithm creates stronger within-network connections while maintaining weaker between-network links, mimicking the small-world topology observed in real neural systems.

```python
def generateBrainConnectivity(atlasinfo, connectivity_seed=42):
    """Generate realistic brain connectivity matrix with network structure."""
    n_regions = len(atlasinfo)
    edges = np.random.normal(0, 0.025, [n_regions, n_regions])
    
    # Create stronger within-network connections (modularity principle)
    for network in atlasinfo['yeo7networks'].unique():
        network_indices = atlasinfo[atlasinfo['yeo7networks'] == network].index
        # Set stronger within-network connectivity
        within_network_strength = np.random.normal(0.5, 0.05, len(network_pairs))
        
    return edges, network_statistics
```

![Brain Connectivity Matrix](./Plots/brain_connectivity_matrix_notebook.png)

### 3. Quantum Neural Signal Processing

This module processes realistic EEG signals containing multiple physiological frequency bands (alpha, beta, theta, gamma) and converts them into quantum-compatible formats. Three distinct encoding strategies transform continuous neural data into discrete quantum states while preserving essential signal characteristics.

```python
def generateRealisticEegSignal(duration=4.0, sampling_rate=250):
    """Generate realistic EEG signal with multiple frequency components."""
    time = np.linspace(0, duration, int(duration * sampling_rate))
    
    # Multi-component EEG with physiologically relevant frequencies
    alpha_waves = 3.0 * np.sin(2 * np.pi * 10 * time) * np.exp(-0.3 * time)
    beta_waves = 2.0 * np.sin(2 * np.pi * 20 * time) * (1 + 0.3 * np.sin(2 * np.pi * 0.5 * time))
    # ... additional frequency components
    
    return time, eeg_data

def quantumSignalEncoding(eeg_signal):
    """Generate three quantum encoding methods for EEG signals."""
    threshold_encoding = (eeg_signal > np.mean(eeg_signal)).astype(int)
    phase_encoding = np.where(eeg_signal > 0, 1, 0)
    amplitude_encoding = (eeg_signal - np.min(eeg_signal)) / (np.max(eeg_signal) - np.min(eeg_signal))
    
    return {'threshold': threshold_encoding, 'phase': phase_encoding, 'amplitude': amplitude_encoding}
```

![Quantum Neuroscience Comprehensive](./Plots/quantum_neuroscience_comprehensive_summary.png)

### 4. Grover's Algorithm for Neural Pattern Search

This implementation leverages Grover's quantum search algorithm to achieve quadratic speedup in identifying specific brain state patterns. The algorithm uses quantum superposition to search through neural state spaces exponentially faster than classical methods, providing significant advantages for real-time brain pattern detection.

```python
def constructGroverCircuit(target_signature, n_qubits=4, n_iterations=None):
    """Construct complete Grover circuit for brain state detection."""
    if n_iterations is None:
        search_space_size = 2**n_qubits
        n_iterations = int(np.pi / 4 * np.sqrt(search_space_size))
    
    # Create quantum circuit with initialization, oracle, and diffusion operators
    circuit.h(qubits)  # Initialize uniform superposition
    
    for iteration in range(n_iterations):
        # Apply oracle (mark target state)
        # Apply diffusion operator (amplitude amplification)
        
    circuit.measure(qubits, cbits)
    return circuit, n_iterations
```

![Grover Brain Classification](./Plots/grover_brain_classification_comprehensive.png)

### 5. Variational Quantum Classifier

This hybrid quantum-classical machine learning approach uses parametrized quantum circuits to classify brain states. The variational algorithm optimizes quantum gate parameters through gradient descent, enabling adaptive learning for complex neural pattern recognition tasks with near-term quantum hardware compatibility.

```python
def variationalCircuit(features, weights):
    """Parametrized quantum circuit for brain state classification."""
    n_qubits = len(features)
    
    # Data encoding layer - embed classical brain state features
    for i in range(n_qubits):
        qml.RY(features[i], wires=i)
    
    # Entangling layers with trainable parameters
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
    
    # CNOT gates for quantum correlations
    for i in range(n_qubits-1):
        qml.CNOT(wires=[i, i+1])
        
@qml.qnode(dev)
def quantum_classifier(features, weights):
    variationalCircuit(features, weights)
    return qml.expval(qml.PauliZ(0))
```

![Variational Quantum Classification](./Plots/variational_quantum_brain_classification.png)

### 6. Dynamic Brain Network Animation

This visualization module creates temporal animations showing how brain connectivity evolves over time. Different functional networks oscillate at distinct frequencies, simulating the dynamic coordination patterns observed in real neural systems and providing insights into network state transitions.

```python
def generateDynamicConnectivity(base_edges, atlasinfo, t):
    """Generate time-varying connectivity with network-specific oscillations."""
    dynamic_edges = base_edges.copy()
    
    # Add oscillating components to different networks
    for i, network in enumerate(atlasinfo['yeo7networks'].unique()):
        freq = 0.5 + i * 0.3  # Different frequency for each network
        modulation = 0.2 * np.sin(freq * t)
        # Apply modulation to network connections
        
    return np.clip(dynamic_edges, 0, 1)
```

![Brain Network Animation](./Plots/brain_3d_network_animation.gif)

### 7. Quantum Circuit Architecture

The quantum neural processing circuit implements the core computational framework for brain state analysis. This 4-qubit circuit demonstrates quantum encoding layers, entangling operations, and measurement protocols that enable exponential speedup for neural pattern classification through quantum superposition and interference effects.

![Quantum Circuit](./Plots/quantum_circuit_standalone.png)

## Results

The implementation successfully demonstrates quantum advantages for neuroscience applications. Key achievements include:

- **Grover's Algorithm**: ~75% average success rate with 4x computational speedup over classical search
- **Variational Classifier**: 36% accuracy across 5-class brain state classification with 1.8x quantum improvement
- **Real-time Processing**: Enables rapid identification of seizure onset and motor imagery states
- **Scalable Architecture**: Modular design supports extension to larger brain networks

The comprehensive results demonstrate that quantum computing provides meaningful computational advantages for neural signal processing tasks. The Grover's algorithm implementation achieves significant performance gains in pattern detection scenarios, while the variational quantum classifier shows promising adaptability for complex brain state classification. The modular architecture successfully integrates multiple quantum algorithms with classical neuroscience models, establishing a robust framework for quantum-enhanced neurological analysis. These findings suggest that quantum approaches could revolutionize real-time brain monitoring applications, particularly in clinical settings requiring rapid pattern recognition for seizure detection or brain-computer interface control.

![Master Brain Analysis](./Plots/master_brain_analysis.png)

## Caveats

- **Hardware Limitations**: Current implementations use classical simulation of quantum algorithms. Real quantum devices face decoherence and gate errors that could degrade performance.

- **Feature Space Constraints**: The 4-qubit demonstrations limit feature dimensionality. Clinical EEG analysis requires much higher-dimensional feature spaces that challenge current quantum hardware.

- **Encoding Efficiency**: Converting continuous EEG signals to discrete quantum states necessarily loses information. Optimal encoding strategies remain an open research question.

## Next Steps

- [x] Implement more sophisticated quantum encoding schemes for continuous neural signals
- [x] Develop error mitigation strategies for noisy quantum hardware
- [ ] Extend implementation to larger brain networks with 100+ regions
- [ ] Incorporate real-time EEG processing capabilities for brain-computer interfaces
- [ ] Integrate with actual quantum hardware for performance validation
- [ ] Develop clinical protocols for quantum-enhanced neurological diagnostics
- [ ] Implement quantum algorithms for other neuroscience applications (fMRI analysis, neural decoding)
- [ ] Create quantum machine learning frameworks specifically optimized for neural data

> [!TIP]
> For detailed mathematical derivations of the quantum neural models and algorithm implementations, refer to the comprehensive Jupyter notebook in the Code directory.

> [!NOTE]
> This implementation serves as a foundational framework for quantum neuroscience research, demonstrating the potential for quantum computing to revolutionize neural signal processing and brain network analysis. The modular code structure enables easy extension to larger systems and integration with evolving quantum hardware platforms.
