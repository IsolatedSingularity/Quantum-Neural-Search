"""Tests for signal encoding correctness."""

import numpy as np

from quantumNeuralSearch.neuralEncoding import (
    calculateNcse,
    calculateShannonEntropy,
    generateRealisticEegSignal,
    quantumSignalEncoding,
    simulateQlif,
)


def test_thresholdEncodingUsesStdScaling() -> None:
    """Threshold encoding should use abs-value std-scaled threshold, not mean."""
    _, eeg = generateRealisticEegSignal(duration=1.0, sampling_rate=100)
    encodings = quantumSignalEncoding(eeg)
    threshold = encodings["threshold"]

    sigma = np.std(eeg)
    expected = (np.abs(eeg) > sigma).astype(int)
    np.testing.assert_array_equal(threshold, expected)


def test_phaseEncodingUsesHilbert() -> None:
    """Phase encoding should use Hilbert transform, not simple sign check."""
    from scipy.signal import hilbert

    _, eeg = generateRealisticEegSignal(duration=1.0, sampling_rate=100)
    encodings = quantumSignalEncoding(eeg)
    phase = encodings["phase"]

    analyticSignal = hilbert(eeg)
    instantaneousPhase = np.angle(analyticSignal)
    expected = (instantaneousPhase > 0).astype(int)
    np.testing.assert_array_equal(phase, expected)


def test_amplitudeEncodingNormalized() -> None:
    """Amplitude encoding should be in [0, 1]."""
    _, eeg = generateRealisticEegSignal(duration=1.0, sampling_rate=100)
    encodings = quantumSignalEncoding(eeg)
    amp = encodings["amplitude"]

    assert np.min(amp) >= 0.0
    assert np.max(amp) <= 1.0
    assert np.isclose(np.min(amp), 0.0)
    assert np.isclose(np.max(amp), 1.0)


def test_shannonEntropyRange() -> None:
    """Shannon entropy for binary windows should be in [0, 1]."""
    _, eeg = generateRealisticEegSignal(duration=2.0, sampling_rate=250)
    entropy = calculateShannonEntropy(eeg)

    assert len(entropy) > 0
    assert np.all(entropy >= 0.0)
    assert np.all(entropy <= 1.0)


def test_ncseRange() -> None:
    """NCSE values should be in [0, 1]."""
    _, eeg = generateRealisticEegSignal(duration=4.0, sampling_rate=250)
    ncse = calculateNcse(eeg)

    assert len(ncse) > 0
    assert np.all(ncse >= 0.0)
    assert np.all(ncse <= 1.0)


def test_qlifExcitationProbability() -> None:
    """QLIF alpha values should stay in [0, 1] (valid probabilities)."""
    rng = np.random.default_rng(42)
    spikeTrain = rng.integers(0, 2, size=200)
    result = simulateQlif(spikeTrain)

    assert np.all(result["alpha"] >= 0.0)
    assert np.all(result["alpha"] <= 1.0)


def test_qlifRespondsToSpikes() -> None:
    """QLIF should produce different traces for all-spike vs no-spike input."""
    allSpikes = np.ones(100, dtype=int)
    noSpikes = np.zeros(100, dtype=int)

    resultSpikes = simulateQlif(allSpikes)
    resultNoSpikes = simulateQlif(noSpikes)

    # Traces should differ
    assert not np.allclose(resultSpikes["alpha"], resultNoSpikes["alpha"])


def test_ncseConstantSignal() -> None:
    """NCSE of a constant signal should yield zero entropy (no information)."""
    constantSignal = np.ones(500)
    ncse = calculateNcse(constantSignal)
    # Constant signal has zero variance, entropy should be 0 or near 0
    assert np.all(ncse <= 0.1), f"NCSE of constant signal too high: max={np.max(ncse):.4f}"


def test_qlifIntegrationMemory() -> None:
    """QLIF alpha[t+1] should depend on alpha[t] (integration memory)."""
    spikeTrain = np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    result = simulateQlif(spikeTrain)
    alpha = result["alpha"]
    # After repeated spikes, alpha should accumulate (not just depend on current spike)
    # Check that alpha during spike sequence is monotonically increasing
    spikeAlphas = alpha[2:5]  # indices 2,3,4 are during spike arrivals
    assert np.all(np.diff(spikeAlphas) >= 0), "QLIF should integrate (accumulate) over spikes"
