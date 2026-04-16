"""Tests for VQC convergence and classical baseline comparison."""

import numpy as np

from quantumNeuralSearch.variationalClassifier import (
    createQuantumClassifier,
    evaluateQuantumClassifier,
    prepareBrainStateTrainingData,
    setupQuantumDevice,
    trainVariationalQuantumClassifier,
)


def test_costDecreasesDuringTraining() -> None:
    """Training cost should decrease over iterations (relaxed: min cost < initial cost)."""
    dev = setupQuantumDevice(4)
    classifier = createQuantumClassifier(dev)
    xTrain, _, yTrain, _, stateNames, _ = prepareBrainStateTrainingData()

    _, costHistory = trainVariationalQuantumClassifier(
        xTrain, yTrain, stateNames, classifier, max_iterations=50, batch_size=10
    )

    assert len(costHistory) == 50
    # The minimum cost seen during training should be lower than the first cost
    initialCost = costHistory[0]
    minCost = min(costHistory)
    assert minCost < initialCost, (
        f"Cost never decreased below initial: initial {initialCost:.4f}, min {minCost:.4f}"
    )


def test_classicalBaselinesPresent() -> None:
    """Evaluation results should include SVM and RF baseline accuracies."""
    dev = setupQuantumDevice(4)
    classifier = createQuantumClassifier(dev)
    xTrain, xTest, yTrain, yTest, stateNames, _ = prepareBrainStateTrainingData()

    weights, _ = trainVariationalQuantumClassifier(
        xTrain, yTrain, stateNames, classifier, max_iterations=10, batch_size=10
    )
    results = evaluateQuantumClassifier(
        xTrain, xTest, yTrain, yTest, weights, classifier, stateNames
    )

    assert "svm_accuracy" in results
    assert "rf_accuracy" in results
    assert results["svm_accuracy"] > 0.0
    assert results["rf_accuracy"] > 0.0


def test_multiQubitMeasurement() -> None:
    """Classifier should return a list/array of expvals, not a scalar."""
    dev = setupQuantumDevice(4)
    classifier = createQuantumClassifier(dev)

    features = np.array([0.5, 0.3, 0.7, 0.1])
    weights = np.random.uniform(0, 2 * np.pi, 12)
    result = classifier(features, weights)

    # Should return 4 expectation values (one per qubit)
    assert hasattr(result, "__len__"), "Classifier should return multiple values"
    assert len(result) == 4, f"Expected 4 expvals, got {len(result)}"
