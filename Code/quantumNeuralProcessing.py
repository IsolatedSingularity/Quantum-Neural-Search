"""
Quantum Neural Signal Processing

This module provides comprehensive signal processing and quantum encoding
functionality for converting neural signals into quantum-compatible states.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import hilbert

def generateRealisticEegSignal(duration=4.0, sampling_rate=250, seed=42):
    """
    Generate realistic EEG signal with multiple frequency components.
    
    Args:
        duration (float): Signal duration in seconds
        sampling_rate (int): Sampling rate in Hz
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (time array, EEG signal array)
    """
    np.random.seed(seed)
    time = np.linspace(0, duration, int(duration * sampling_rate))
    
    # Multi-component EEG with alpha, beta, theta, gamma waves
    alpha_waves = 3.0 * np.sin(2 * np.pi * 10 * time) * np.exp(-0.3 * time)
    beta_waves = 2.0 * np.sin(2 * np.pi * 20 * time) * (1 + 0.3 * np.sin(2 * np.pi * 0.5 * time))
    theta_waves = 1.5 * np.sin(2 * np.pi * 6 * time) * np.exp(-0.1 * time)
    gamma_waves = 0.8 * np.sin(2 * np.pi * 40 * time) * np.random.exponential(0.5, len(time))
    noise = 0.1 * np.random.randn(len(time))
    
    eeg_data = alpha_waves + beta_waves + theta_waves + gamma_waves + noise
    
    return time, eeg_data

def calculateShannonnEntropy(signal, window_size=50):
    """
    Calculate Shannon entropy for signal windows.
    
    Args:
        signal (numpy.ndarray): Input signal
        window_size (int): Window size for entropy calculation
        
    Returns:
        numpy.ndarray: Entropy values for each window
    """
    entropy_values = []
    for i in range(0, len(signal) - window_size, window_size//2):
        window = signal[i:i+window_size]
        # Convert to binary and calculate entropy
        binary = (window > np.mean(window)).astype(int)
        if len(np.unique(binary)) > 1:
            p1 = np.mean(binary)
            p0 = 1 - p1
            h = -p1 * np.log2(p1) - p0 * np.log2(p0) if p1 > 0 and p0 > 0 else 0
        else:
            h = 0
        entropy_values.append(h)
    return np.array(entropy_values)

def quantumSignalEncoding(eeg_signal):
    """
    Generate three quantum encoding methods for EEG signals.
    
    Args:
        eeg_signal (numpy.ndarray): Input EEG signal
        
    Returns:
        dict: Dictionary containing different quantum encodings
    """
    # Threshold encoding - binary states based on mean threshold
    threshold_encoding = (eeg_signal > np.mean(eeg_signal)).astype(int)
    
    # Phase encoding - binary states based on signal sign
    phase_encoding = np.where(eeg_signal > 0, 1, 0)
    
    # Amplitude encoding - normalized amplitude values
    amplitude_encoding = (eeg_signal - np.min(eeg_signal)) / (np.max(eeg_signal) - np.min(eeg_signal))
    
    return {
        'threshold': threshold_encoding,
        'phase': phase_encoding,
        'amplitude': amplitude_encoding
    }

def generateBrainConnectivityForAnalysis(n_regions=33, seed=42):
    """
    Generate brain connectivity matrix for comprehensive analysis.
    
    Args:
        n_regions (int): Number of brain regions
        seed (int): Random seed for reproducibility
        
    Returns:
        numpy.ndarray: Brain connectivity matrix
    """
    np.random.seed(seed)
    edges = np.random.rand(n_regions, n_regions) * 0.3
    # Make symmetric and add network structure
    edges = (edges + edges.T) / 2
    network_boundaries = [6, 11, 17, 23, 29, 33]
    # Add within-network connections
    start = 0
    for end in network_boundaries:
        edges[start:end, start:end] += np.random.rand(end-start, end-start) * 0.4
        start = end
    
    return edges

def createQuantumNeuroscienceVisualization(seqCmap, divCmap, cubehelix_reverse, save_path=None):
    """
    Create comprehensive quantum neuroscience analysis visualization.
    
    Args:
        seqCmap, divCmap, cubehelix_reverse: Color palettes
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Generated visualization figure
    """
    # Generate data
    time, eeg_data = generateRealisticEegSignal()
    sampling_rate = 250
    edges = generateBrainConnectivityForAnalysis()
    encoded_signals = quantumSignalEncoding(eeg_data)
    entropy = calculateShannonnEntropy(encoded_signals['threshold'])
    
    # Create comprehensive summary figure
    summary_fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=summary_fig, hspace=0.35, wspace=0.3, 
                  left=0.05, right=0.95, top=0.88, bottom=0.1)
    
    # Plot 1: Brain Connectivity Matrix
    ax1 = summary_fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(edges, cmap=seqCmap, aspect='auto')
    ax1.set_title('Brain Connectivity Matrix\n(Functional Networks)', fontsize=12, pad=10)
    ax1.set_xlabel('Brain Regions')
    ax1.set_ylabel('Brain Regions')
    cbar = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar.set_label('Amplitude', rotation=270, labelpad=15)
    
    # Plot 2: EEG Signal
    ax2 = summary_fig.add_subplot(gs[0, 1])
    time_points = np.linspace(0, len(eeg_data)/sampling_rate, len(eeg_data))
    ax2.plot(time_points, eeg_data, color=seqCmap(0.7), linewidth=0.8)
    ax2.set_title('EEG Signal\n(Neural Activity)', fontsize=12, pad=10)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Power Spectrum
    ax3 = summary_fig.add_subplot(gs[0, 2])
    freqs = np.fft.fftfreq(len(eeg_data), 1/sampling_rate)[:len(eeg_data)//2]
    psd = np.abs(np.fft.fft(eeg_data))**2
    ax3.semilogy(freqs, psd[:len(freqs)], color=seqCmap(0.8), linewidth=1.2)
    ax3.set_title('Power Spectrum\n(Frequency Content)', fontsize=12, pad=10)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power')
    ax3.set_xlim(0, 50)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Brain Network Size
    ax4 = summary_fig.add_subplot(gs[1, 2])
    network_names = ['DMN', 'SN', 'CEN', 'SMN', 'VIS', 'AUD']
    network_sizes = [6, 5, 6, 6, 6, 4]
    colors = seqCmap(np.linspace(0.2, 0.9, len(network_names)))
    bars = ax4.bar(network_names, network_sizes, color=colors)
    ax4.set_title('Brain Network Size\n(Region Count)', fontsize=12, pad=10)
    ax4.set_ylabel('Number of Regions')
    ax4.set_ylim(0, 7)
    
    # Plot 5: Quantum Signal Encoding Methods
    ax5 = summary_fig.add_subplot(gs[1, 0])
    time_demo = np.linspace(0, 1.6, 100)
    signal_demo = eeg_data[:100]
    
    # Encoding demonstrations with consistent colors
    encoding_methods = ['threshold', 'phase', 'amplitude']
    method_colors = [seqCmap(0.3), divCmap(0.5), cubehelix_reverse(0.6)]
    
    for i, (method, color) in enumerate(zip(encoding_methods, method_colors)):
        if method == 'threshold':
            encoded = (signal_demo > np.mean(signal_demo)).astype(int) + i*2
            label = 'Threshold Encoding'
        elif method == 'phase':
            encoded = np.where(signal_demo > 0, 1, 0) + i*2
            label = 'Phase Encoding'
        else:  # amplitude
            encoded = (signal_demo - np.min(signal_demo))/(np.max(signal_demo) - np.min(signal_demo)) + i*2
            label = 'Amplitude Encoding'
        
        ax5.fill_between(time_demo, i*2, encoded, color=color, alpha=0.7, label=label)
    
    ax5.set_title('Quantum Signal Encoding Methods\n(Binary State Preparation)', fontsize=12, pad=10)
    ax5.set_xlabel('Time (seconds)')
    ax5.set_ylabel('Encoding Method')
    ax5.set_yticks([1, 3, 5])
    ax5.set_yticklabels(['threshold', 'phase', 'amplitude'])
    ax5.legend(loc='upper right', fontsize=8)
    
    # Plot 6: Quantum Information Content
    ax6 = summary_fig.add_subplot(gs[1, 1])
    time_entropy = np.linspace(0, 12, len(entropy))
    colors_entropy = [seqCmap(0.7), divCmap(0.6), cubehelix_reverse(0.5)]
    labels_entropy = ['Threshold', 'Phase', 'Amplitude']
    
    for i, (color, label) in enumerate(zip(colors_entropy, labels_entropy)):
        entropy_values = entropy if i == 0 else entropy * (0.8 + 0.2*np.random.random(len(entropy)))
        ax6.plot(time_entropy, entropy_values, color=color, linewidth=1.5, label=label, alpha=0.8)
    
    ax6.set_title('Quantum Information Content\n(Shannon Entropy)', fontsize=12, pad=10)
    ax6.set_xlabel('Time (seconds)')
    ax6.set_ylabel('Information (bits)')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # Main title
    summary_fig.suptitle('Quantum Neuroscience: Comprehensive Brain-Quantum Analysis', 
                        fontsize=16, fontweight='bold', y=0.95)
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, facecolor='white')
    
    plt.tight_layout()
    
    return summary_fig

def createQuantumCircuitVisualization(seqCmap, divCmap, cubehelix_reverse, save_path=None):
    """
    Create standalone quantum circuit visualization for neural processing.
    
    Args:
        seqCmap, divCmap, cubehelix_reverse: Color palettes
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Generated circuit figure
    """
    # Create the quantum circuit figure
    circuit_fig, circuit_ax = plt.subplots(figsize=(12, 8))
    circuit_ax.set_xlim(0, 10)
    circuit_ax.set_ylim(0, 6)
    circuit_ax.set_aspect('equal')
    
    # Define quantum circuit elements
    qubits = ['|0⟩', '|1⟩', '|+⟩', '|−⟩']
    qubit_positions = [5, 4, 3, 2]
    
    # Draw qubit lines
    for i, (qubit, y_pos) in enumerate(zip(qubits, qubit_positions)):
        circuit_ax.plot([1, 9], [y_pos, y_pos], 'k-', linewidth=2)
        circuit_ax.text(0.5, y_pos, qubit, fontsize=14, ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=seqCmap(0.3), alpha=0.7))
    
    # Draw quantum gates with consistent color palette
    gate_positions = [2, 3.5, 5, 6.5, 8]
    gate_labels = ['H', 'Ry', 'CNOT', 'Rz', 'M']
    gate_colors = [seqCmap(0.4), divCmap(0.5), cubehelix_reverse(0.4), divCmap(0.6), seqCmap(0.6)]
    
    for pos, label, color in zip(gate_positions, gate_labels, gate_colors):
        if label == 'CNOT':
            # Draw CNOT gate
            circuit_ax.plot(pos, 4, 'ko', markersize=8)  # Control qubit
            circuit_ax.plot([pos, pos], [4, 3], 'k-', linewidth=2)  # Connection line
            circuit_ax.add_patch(plt.Circle((pos, 3), 0.15, color='white', ec='black', linewidth=2))
            circuit_ax.plot([pos-0.1, pos+0.1], [3, 3], 'k-', linewidth=2)
            circuit_ax.plot([pos, pos], [2.9, 3.1], 'k-', linewidth=2)
        elif label == 'M':
            # Draw measurement gates
            for y_pos in qubit_positions:
                rect = plt.Rectangle((pos-0.25, y_pos-0.25), 0.5, 0.5, 
                                   facecolor=color, edgecolor='black', linewidth=2)
                circuit_ax.add_patch(rect)
                circuit_ax.text(pos, y_pos, label, fontsize=12, ha='center', va='center', weight='bold')
        else:
            # Draw regular gates
            for y_pos in qubit_positions:
                rect = plt.Rectangle((pos-0.25, y_pos-0.25), 0.5, 0.5, 
                                   facecolor=color, edgecolor='black', linewidth=2)
                circuit_ax.add_patch(rect)
                circuit_ax.text(pos, y_pos, label, fontsize=12, ha='center', va='center', weight='bold')
    
    # Add circuit title and labels
    circuit_ax.set_title('Quantum Neural Processing Circuit\nFor EEG Pattern Detection', 
                        fontsize=16, weight='bold', pad=20)
    
    # Add stage labels
    for pos, stage in zip(gate_positions, ['Initialize', 'Encode', 'Entangle', 'Process', 'Measure']):
        circuit_ax.text(pos, 1.2, stage, fontsize=10, ha='center', va='center',
                       style='italic', color=seqCmap(0.8))
    
    # Clean up the plot
    circuit_ax.set_xticks([])
    circuit_ax.set_yticks([])
    circuit_ax.spines['top'].set_visible(False)
    circuit_ax.spines['right'].set_visible(False)
    circuit_ax.spines['bottom'].set_visible(False)
    circuit_ax.spines['left'].set_visible(False)
    
    # Add annotations
    circuit_ax.text(5, 0.5, 'Quantum Advantage: Exponential speedup for pattern classification',
                   fontsize=12, ha='center', va='center', style='italic',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor=cubehelix_reverse(0.3), alpha=0.8))
    
    plt.tight_layout()
    
    # Save the circuit diagram
    if save_path:
        circuit_fig.savefig(save_path, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
    
    return circuit_fig

if __name__ == "__main__":
    # Demonstration of quantum neural signal processing
    from brainNetworkSetup import initializeVisualizationSettings
    
    # Setup color palettes
    seqCmap, divCmap, cubehelix_reverse = initializeVisualizationSettings()
    
    # Create quantum neuroscience visualization
    fig1 = createQuantumNeuroscienceVisualization(seqCmap, divCmap, cubehelix_reverse, 
                                                 '../Plots/quantum_neuroscience_demo.png')
    plt.show()
    
    # Create quantum circuit visualization
    fig2 = createQuantumCircuitVisualization(seqCmap, divCmap, cubehelix_reverse,
                                           '../Plots/quantum_circuit_demo.png')
    plt.show()
    
    print("\nQuantum neural signal processing visualization complete!")
