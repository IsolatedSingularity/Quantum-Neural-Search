# Quantum Neural Search
###### Grover search and variational quantum circuits for 4-qubit EEG pattern classification, built with Qiskit and PennyLane. All data is synthetically generated for algorithm validation.

<p align="center">
  <a href="https://github.com/IsolatedSingularity/Quantum-Neural-Search/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/IsolatedSingularity/Quantum-Neural-Search/ci.yml?branch=main&label=CI&logo=github" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache-2.0"></a>
</p>

![Master Brain Analysis](./Plots/master_brain_analysis.png)

## Objective

This repository implements quantum algorithms for neuroscience applications: exact pattern search using Grover's algorithm and adaptive classification using variational quantum circuits. The system operates on a 33-region brain atlas with 4-qubit encoding, using synthetic EEG signals with physiologically realistic frequency bands (alpha, beta, theta, gamma).

Neural state spaces scale as $2^N$ for $N$ feature qubits. Grover's algorithm searches this space in $O(\sqrt{N})$ oracle calls instead of $O(N)$, while the variational classifier learns decision boundaries through parametrized quantum gates optimized via gradient descent.

**Goal:** Demonstrate quantum search and classification on synthetic neural data, compare against classical baselines (SVM, Random Forest), and provide a modular codebase for quantum neuroscience experiments on near-term hardware.

## Theoretical Background

### Neural Dynamics and Quantum Encoding

Biological neural networks are modeled using the Leaky Integrate-and-Fire (LIF) neuron framework:

$$\tau \frac{dV}{dt} = -V(t) + I(t)$$

where $V(t)$ represents membrane potential, $\tau$ is the membrane time constant, and $I(t)$ represents synaptic input. For network-level analysis, this extends to coupled systems with connectivity matrices:

$$\frac{d\mathbf{V}}{dt} = -\frac{1}{\tau}\mathbf{V} + \mathbf{W}\mathbf{s}(t) + \mathbf{I}_{ext}(t)$$

The Quantum Leaky Integrate-and-Fire (QLIF) model represents neural excitation through qubit state probabilities:

$$\alpha[t+1] = \sin^2\left(\frac{(\theta + \varphi[t])X[t+1] + (\gamma[t] + \varphi[t])(1-X[t+1])}{2}\right)$$

where $\alpha[t]$ is the excited-state probability, $\theta$ controls spike rotation angles, and $\gamma[t]$ models quantum decoherence via exponential $T_2$ decay.

### Quantum Algorithm Formulations

Grover's search uses amplitude amplification with a multi-controlled phase gate oracle:

$$|\psi_{final}\rangle = G^{k} |s\rangle, \quad k = \left\lfloor \frac{\pi}{4\arcsin(1/\sqrt{N})} \right\rceil$$

where $G = -U_s U_f$ combines the oracle $U_f$ (marks target brain states via `mcp`) and diffusion operator $U_s$ (amplifies marked amplitudes).

Variational quantum circuits classify brain states through parametrized gates:

$$\langle \hat{H} \rangle_{\boldsymbol{\theta}} = \langle \psi_{data} | U^{\dagger}(\boldsymbol{\theta}) \hat{H} U(\boldsymbol{\theta}) | \psi_{data} \rangle$$

where $U(\boldsymbol{\theta})$ is optimized via gradient descent. Multi-qubit measurement returns one expectation value per qubit for 5-class discrimination.

### Signal Encoding Strategies

Three encoding methods convert continuous EEG signals to quantum-compatible discrete states:

Threshold encoding creates binary representations using standard-deviation scaling:
```math
b[n] = \begin{cases} 
1 & \text{if } |x[n]| > \theta_{factor} \cdot \sigma(x) \\ 
0 & \text{otherwise} 
\end{cases}
```

Phase encoding uses the Hilbert transform to extract instantaneous phase:
$$\phi[n] = \arg(\mathcal{H}(x[n]))$$

Amplitude encoding normalizes magnitudes to $[0, 1]$:
$$a[n] = \frac{x[n] - x_{min}}{x_{max} - x_{min}}$$

Neural complexity is quantified using Normalized Corrected Shannon Entropy (NCSE):
$$NCSE(L,\Psi) = \frac{CSE(L,\Psi)}{CSE_{max}(L,\Psi)}$$

where $L$ is symbolic word length, $\Psi$ is the symbolic sequence, and $CSE$ includes Miller-Madow bias correction.

## Code Functionality

### 1. Brain Network Setup and Atlas Creation

Creates a 33-region brain atlas organized across 6 major functional networks with anatomically plausible 3D coordinates in MNI space.

```python
from quantumNeuralSearch.brainAtlas import createBrainAtlas

atlasinfo, coords, nodesDf = createBrainAtlas()
# Returns atlas DataFrame, 3D coordinates, and node metadata
```

![Brain Networks Multiview](./Plots/brain_networks_multiview_notebook.png)

### 2. Brain Connectivity Matrix Generation

Generates biologically realistic connectivity matrices with stronger within-network connections and weaker between-network links, mimicking small-world topology.

```python
from quantumNeuralSearch.brainConnectivity import generateBrainConnectivity

edges, stats = generateBrainConnectivity(atlasinfo, connectivity_seed=42)
```

![Brain Connectivity Matrix](./Plots/brain_connectivity_matrix_notebook.png)

### 3. Quantum Neural Signal Processing

Generates synthetic EEG signals with multiple physiological frequency bands and encodes them using three strategies: Hilbert-transform phase encoding, std-scaled threshold encoding, and amplitude normalization. Includes QLIF neuron simulation and NCSE computation.

```python
from quantumNeuralSearch.neuralEncoding import (
    generateRealisticEegSignal,
    quantumSignalEncoding,
    simulateQlif,
    calculateNcse,
)

time, eeg = generateRealisticEegSignal(duration=4.0, sampling_rate=250)
encodings = quantumSignalEncoding(eeg)
# encodings['phase']     - Hilbert transform binary encoding
# encodings['threshold'] - |x| > sigma threshold encoding
# encodings['amplitude'] - min-max normalized to [0, 1]

qlifResult = simulateQlif(spikeTrain)      # QLIF neuron dynamics
ncseValues = calculateNcse(eeg)             # Normalized Corrected Shannon Entropy
```

![Quantum Neuroscience Comprehensive](./Plots/quantum_neuroscience_comprehensive_summary.png)

### 4. Grover's Algorithm for Neural Pattern Search

Uses Grover's quantum search to find specific brain state patterns from a 16-state search space. The oracle marks target states via a multi-controlled phase gate (`circuit.mcp`), and the exact iteration formula ensures optimal amplification.

```python
from quantumNeuralSearch.groversSearch import constructGroverCircuit, initializeGroverSearch

brainSignatures, searchParams, simulator = initializeGroverSearch()
circuit, iterations = constructGroverCircuit(
    target_signature=[1, 0, 1, 1],  # motor_left
    n_qubits=4
)
# iterations = 3 (optimal for N=16, 1 target)
job = simulator.run(circuit, shots=2048)
```

![Grover Brain Classification](./Plots/grover_brain_classification_comprehensive.png)

### 5. Variational Quantum Classifier

Hybrid quantum-classical classifier using parametrized quantum circuits with multi-qubit measurement. Compares against SVM and Random Forest baselines on the same data.

```python
from quantumNeuralSearch.variationalClassifier import (
    setupQuantumDevice,
    createQuantumClassifier,
    prepareBrainStateTrainingData,
    trainVariationalQuantumClassifier,
    evaluateQuantumClassifier,
)

dev = setupQuantumDevice(4)
classifier = createQuantumClassifier(dev)
xTrain, xTest, yTrain, yTest, stateNames, scaler = prepareBrainStateTrainingData()
weights, costHistory = trainVariationalQuantumClassifier(
    xTrain, yTrain, stateNames, classifier, max_iterations=100
)
results = evaluateQuantumClassifier(xTrain, xTest, yTrain, yTest, weights, classifier, stateNames)
# results includes: test_accuracy, svm_accuracy, rf_accuracy, svm_comparison, rf_comparison
```

![Variational Quantum Classification](./Plots/variational_quantum_brain_classification.png)

### 6. Dynamic Brain Network Animation

Temporal animations showing brain connectivity evolution, with network-specific oscillation frequencies simulating dynamic coordination patterns.

```python
from quantumNeuralSearch.dynamicStates import generateDynamicConnectivity

dynamicEdges = generateDynamicConnectivity(baseEdges, atlasinfo, t=0.5)
```

![Brain Network Animation](./Plots/brain_3d_network_animation.gif)

## Installation

```bash
git clone https://github.com/IsolatedSingularity/Quantum-Neural-Search.git
cd Quantum-Neural-Search
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .
```

Requires Python 3.10+. Core dependencies: Qiskit >= 1.0, PennyLane >= 0.35, NumPy, SciPy, scikit-learn, matplotlib.

## Project Structure

```
Quantum-Neural-Search/
  pyproject.toml
  LICENSE
  README.md
  .github/workflows/ci.yml
  quantumNeuralSearch/
    __init__.py
    groversSearch.py          # Grover oracle + search
    variationalClassifier.py  # VQC + classical baselines
    neuralEncoding.py         # Signal encoding, QLIF, NCSE
    brainAtlas.py             # 33-region brain atlas
    brainConnectivity.py      # Connectivity matrix generation
    dynamicStates.py          # Dynamic brain animation
    visualization/
      __init__.py
      groversPlots.py
      variationalPlots.py
      brainPlots.py
      masterAnalysis.py
  tests/
    testGroverOracle.py       # Oracle correctness, iteration count
    testEncodings.py          # Signal encoding, entropy, QLIF
    testVqcConvergence.py     # VQC training, baselines, measurement
  Plots/
```

## Results

On synthetic 4-qubit EEG data with 5 brain states (motor left/right, seizure onset, rest, cognitive load):

- **Grover's Search**: ~75% average success rate for target state amplification (optimal 3 iterations for N=16). Speedup is $O(\sqrt{N})$ vs $O(N)$ classical brute-force enumeration.
- **Variational Classifier**: ~36% accuracy on 5 classes (20% random baseline). Classical baselines (SVM, Random Forest) are evaluated on the same train/test split for direct comparison.
- **14 tests** covering oracle correctness, encoding fidelity, and VQC convergence.

The 4-qubit feature space is intentionally small for demonstration. Scaling to clinically relevant dimensionality requires hardware beyond current NISQ devices.

## Caveats

- **Simulation only**: All quantum circuits run on classical simulators (Qiskit Aer, PennyLane default.qubit). Real hardware introduces decoherence and gate errors.
- **Synthetic data**: No clinical or patient data is used. EEG signals are generated with physiologically plausible frequency components but are not real recordings.
- **Feature space**: 4 qubits encode only 16 states. Clinical EEG analysis requires much higher dimensionality.
- **Encoding loss**: Converting continuous EEG to discrete quantum states necessarily discards information. Optimal encoding remains an open problem.

## Future Goals

- [ ] Extend to larger brain networks with 100+ regions
- [ ] Incorporate real-time EEG processing for brain-computer interfaces
- [ ] Validate on actual quantum hardware
- [ ] Develop error mitigation strategies for noisy quantum hardware
- [ ] Implement more sophisticated encoding schemes for continuous neural signals
- [ ] Create quantum ML frameworks optimized for neural data

## Acknowledgments

Built by [Jeffrey Morais](https://ichor.pages.dev/). Brain atlas coordinates follow the MNI152 standard. Quantum circuits use [Qiskit](https://qiskit.org/) and [PennyLane](https://pennylane.ai/).
