"""
Quantum Neural Search: quantum algorithms for neuroscience applications.

Grover's search on brain networks, variational quantum circuits for EEG
classification. 33-region brain atlas, QLIF model, signal encoding.
"""

from quantumNeuralSearch.brainAtlas import createBrainAtlas, initializeVisualizationSettings
from quantumNeuralSearch.brainConnectivity import generateBrainConnectivity
from quantumNeuralSearch.groversSearch import (
    constructGroverCircuit,
    executeGroverClassification,
    initializeGroverSearch,
)
from quantumNeuralSearch.neuralEncoding import (
    calculateNcse,
    calculateShannonEntropy,
    generateRealisticEegSignal,
    quantumSignalEncoding,
    simulateQlif,
)
from quantumNeuralSearch.variationalClassifier import (
    createQuantumClassifier,
    evaluateQuantumClassifier,
    prepareBrainStateTrainingData,
    setupQuantumDevice,
    trainVariationalQuantumClassifier,
)

__all__ = [
    "createBrainAtlas",
    "initializeVisualizationSettings",
    "generateBrainConnectivity",
    "initializeGroverSearch",
    "constructGroverCircuit",
    "executeGroverClassification",
    "setupQuantumDevice",
    "createQuantumClassifier",
    "prepareBrainStateTrainingData",
    "trainVariationalQuantumClassifier",
    "evaluateQuantumClassifier",
    "generateRealisticEegSignal",
    "simulateQlif",
    "calculateShannonEntropy",
    "calculateNcse",
    "quantumSignalEncoding",
]
