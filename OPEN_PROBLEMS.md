# Open Problems: Quantum-Neural-Search

This document catalogs open problems, algorithmic defects, and architectural gaps for the **Quantum-Neural-Search** EEG brain-network classification codebase (`Code/groversNeuralSearch.py`, `Code/variationalQuantumClassifier.py`, `Code/quantumNeuralProcessing.py`).

---

## 1. Algorithmic & Implementation Problems

- **4-Qubit Grover Oracle and Diffusion Operator Correctness (`Q1, Q2`)**
  - **Problem**: `Code/groversNeuralSearch.py` incorrectly uses data qubit 0 as the Toffoli (`CCX`) target in the 4-qubit oracle and diffusion blocks, corrupting computational state amplitudes rather than applying a 4-controlled phase flip. Furthermore, the 16-line block is duplicated verbatim.
  - **Context**: Requires implementing an ancilla-assisted multi-controlled X (`mcx`) or multi-controlled Z phase shift.
- **5-Class Variational Quantum Classifier (VQC) Measurement Strategy (`Q5`)**
  - **Problem**: `Code/variationalQuantumClassifier.py` measures a single scalar expectation value `qml.expval(qml.PauliZ(0))` in the range $[-1, 1]$, which is mathematically incapable of linearly separating 5 distinct EEG classification classes.
  - **Context**: Requires multi-qubit Pauli expectation vector readouts or a One-vs-Rest (OvR) classifier ensemble.
- **Hilbert Transform Phase Encoding (`Q3`)**
  - **Problem**: While `README.md` specifies analytic signal Hilbert transform phase encoding, `Code/quantumNeuralProcessing.py` implements a basic sign threshold (`np.where(eeg_signal > 0, 1, 0)`), leaving `scipy.signal.hilbert` unused.
- **Quantum Leaky Integrate-and-Fire (QLIF) Neuron Implementation (`Q10`)**
  - **Problem**: `README.md` documents a QLIF neural model with mathematical formulations, but zero implementing code exists in the repository. Requires building a PennyLane QNode-based QLIF neuron.
- **Normalized Complexity Shannon Entropy (NCSE) Metric (`Q4, Q11`)**
  - **Problem**: `calculateShannonnEntropy` contains a typo and computes basic binary entropy rather than the documented multi-scale NCSE network complexity metric.

---

## 2. Bugs & Unresolved Issues

- **Optimizer Re-creation in Training Loop (`Q12`)**
  - **Problem**: `Code/variationalQuantumClassifier.py` re-initializes `GradientDescentOptimizer` inside the epoch training loop, destroying optimizer momentum and state history on every iteration.
- **Silent Classical Fallback Model (`Q13`)**
  - **Problem**: When PennyLane is unavailable, `quantum_classifier_sim` falls back to `np.tanh(dot_product)` without warning the user that quantum simulation has been disabled.
- **Misleading Quantum Advantage Metric (`Q6`)**
  - **Problem**: Reporting a 1.8x "quantum advantage" by dividing 36% VQC test accuracy by a 20% random guess baseline without benchmarking against classical supervised ML classifiers (SVM, Random Forest).

---

## 3. Theoretical & Scientific Problems

- **EEG Feature Discretization and Grover Search Selectivity**
  - **Problem**: Establishing analytical bounds on amplitude amplification efficiency when continuous multichannel EEG bandpower signals are quantized into 4-qubit oracle states under noisy experimental conditions.

---

## 4. Code Maintenance & Refactoring Opportunities

- **Python Package Modularization (`Q15, D8`)**
  - **Opportunity**: Bare relative imports (`from brainNetworkSetup import ...`) restrict execution to the `Code/` working directory. Transforming `Code/` into a structured Python package (`quantumNeuralSearch/`) with explicit module imports will improve usability and testability.
