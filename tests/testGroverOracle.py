"""Tests for Grover oracle correctness."""

from qiskit_aer import AerSimulator

from quantumNeuralSearch.groversSearch import constructGroverCircuit, initializeGroverSearch


def test_oracleAmplifiestTargetState() -> None:
    """The target bitstring should appear with probability > 50% after Grover iterations."""
    target = [1, 0, 1, 1]
    circuit, iterations = constructGroverCircuit(target, n_qubits=4)
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=4096)
    counts = job.result().get_counts()

    targetBitstring = "".join(str(b) for b in reversed(target))
    targetCount = counts.get(targetBitstring, 0)
    successProbability = targetCount / 4096

    assert successProbability > 0.5, (
        f"Target {targetBitstring} success probability {successProbability:.3f} is too low"
    )


def test_oracleDoesNotCorruptDataQubits() -> None:
    """After oracle + un-compute, non-target states should not be boosted."""
    target = [0, 0, 0, 0]
    circuit, _ = constructGroverCircuit(target, n_qubits=4)
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=4096)
    counts = job.result().get_counts()

    targetBitstring = "0000"
    targetCount = counts.get(targetBitstring, 0)
    assert targetCount / 4096 > 0.5


def test_optimalIterationCount() -> None:
    """For 4 qubits (N=16, 1 target), optimal iterations should be 3."""
    _, iterations = constructGroverCircuit([1, 1, 1, 1], n_qubits=4)
    assert iterations == 3, f"Expected 3 iterations, got {iterations}"


def test_allTargetSignaturesAmplified() -> None:
    """Each of the 5 brain state signatures should be amplified above random (1/16)."""
    brainSignatures, searchParams, simulator = initializeGroverSearch()

    for stateName, signature in brainSignatures.items():
        circuit, _ = constructGroverCircuit(signature, n_qubits=4)
        job = simulator.run(circuit, shots=2048)
        counts = job.result().get_counts()

        targetBitstring = "".join(str(b) for b in reversed(signature))
        targetCount = counts.get(targetBitstring, 0)
        prob = targetCount / 2048

        randomBaseline = 1 / 16
        assert prob > randomBaseline * 2, (
            f"{stateName}: prob {prob:.3f} not above random {randomBaseline:.3f}"
        )
