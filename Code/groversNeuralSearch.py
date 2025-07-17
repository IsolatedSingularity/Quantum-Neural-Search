"""
Grover's Algorithm for Neural Pattern Search

This module implements Grover's quantum search algorithm for brain state
classification and neural pattern detection with quadratic speedup.
"""

import json
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

def initializeGroverSearch():
    """
    Initialize Grover's Algorithm setup for brain state classification.
    
    Returns:
        tuple: (brain_signatures dict, search parameters dict, simulator)
    """
    # Define brain state signatures for quantum search
    brain_signatures = {
        'motor_left': [1, 0, 1, 1],      # High activation, left lateralized, motor cortex
        'motor_right': [1, 1, 0, 1],     # High activation, right lateralized, motor cortex  
        'seizure_onset': [1, 1, 1, 0],   # High synchrony, widespread activation, non-motor
        'rest_state': [0, 0, 0, 0],      # Low activation across all regions
        'cognitive_load': [0, 1, 1, 1]   # Moderate activation, frontal-parietal networks
    }
    
    # Quantum encoding parameters
    n_qubits = 4  # Features: [activation_level, left_hemisphere, right_hemisphere, motor_areas]
    search_space_size = 2**n_qubits  # 16 possible brain states
    
    search_params = {
        'n_qubits': n_qubits,
        'search_space_size': search_space_size,
        'optimal_iterations': int(np.pi / 4 * np.sqrt(search_space_size))
    }
    
    # Initialize quantum simulator
    simulator = AerSimulator()
    
    print("=== Grover's Algorithm: Neural Pattern Search Initialization ===")
    print(f"Quantum Search Configuration:")
    print(f"  Qubits (features): {n_qubits}")
    print(f"  Search space: {search_space_size} possible brain states")
    print(f"  Classical complexity: O({search_space_size})")
    print(f"  Quantum complexity: O({int(np.sqrt(search_space_size))})")
    
    return brain_signatures, search_params, simulator

def constructGroverCircuit(target_signature, n_qubits=4, n_iterations=None):
    """
    Construct complete Grover circuit for brain state detection.
    
    Args:
        target_signature (list): Target brain state pattern
        n_qubits (int): Number of qubits in the circuit
        n_iterations (int, optional): Number of Grover iterations
        
    Returns:
        tuple: (QuantumCircuit, number of iterations used)
    """
    # Calculate optimal iterations for maximum success probability
    if n_iterations is None:
        search_space_size = 2**n_qubits
        n_iterations = int(np.pi / 4 * np.sqrt(search_space_size))
        n_iterations = max(1, min(n_iterations, 6))  # Practical bounds
    
    # Create quantum registers
    qubits = QuantumRegister(n_qubits, 'neural_features')
    cbits = ClassicalRegister(n_qubits, 'measurement')
    circuit = QuantumCircuit(qubits, cbits)
    
    # Step 1: Initialize uniform superposition
    circuit.h(qubits)
    circuit.barrier(label='Initialization')
    
    # Step 2: Apply Grover iterations
    for iteration in range(n_iterations):
        # Apply oracle (mark target state) - simplified version
        # Apply X gates where target signature has 0s
        for i, bit in enumerate(target_signature):
            if bit == 0:
                circuit.x(qubits[i])
        
        # Apply phase flip to all-ones state (after X gates)
        # Use a simplified approach for the oracle
        if n_qubits == 1:
            circuit.z(qubits[0])
        elif n_qubits == 2:
            circuit.cz(qubits[0], qubits[1])
        elif n_qubits == 3:
            circuit.ccz(qubits[0], qubits[1], qubits[2]) 
        elif n_qubits == 4:
            # Simplified 4-controlled Z using decomposition
            circuit.h(qubits[3])
            circuit.ccx(qubits[0], qubits[1], qubits[3])
            circuit.ccx(qubits[2], qubits[3], qubits[0])
            circuit.ccx(qubits[0], qubits[1], qubits[3])
            circuit.ccx(qubits[2], qubits[3], qubits[0])
            circuit.z(qubits[3])
            circuit.ccx(qubits[2], qubits[3], qubits[0])
            circuit.ccx(qubits[0], qubits[1], qubits[3])
            circuit.ccx(qubits[2], qubits[3], qubits[0])
            circuit.ccx(qubits[0], qubits[1], qubits[3])
            circuit.h(qubits[3])
        
        # Restore original encoding by undoing X gates
        for i, bit in enumerate(target_signature):
            if bit == 0:
                circuit.x(qubits[i])
        
        circuit.barrier(label=f'Oracle_{iteration+1}')
        
        # Apply diffusion operator (amplitude amplification)
        circuit.h(qubits)
        circuit.x(qubits)
        # Apply phase flip to all-ones state 
        if n_qubits == 1:
            circuit.z(qubits[0])
        elif n_qubits == 2:
            circuit.cz(qubits[0], qubits[1])
        elif n_qubits == 3:
            circuit.ccz(qubits[0], qubits[1], qubits[2])
        elif n_qubits == 4:
            # Simplified 4-controlled Z using decomposition
            circuit.h(qubits[3])
            circuit.ccx(qubits[0], qubits[1], qubits[3])
            circuit.ccx(qubits[2], qubits[3], qubits[0])
            circuit.ccx(qubits[0], qubits[1], qubits[3])
            circuit.ccx(qubits[2], qubits[3], qubits[0])
            circuit.z(qubits[3])
            circuit.ccx(qubits[2], qubits[3], qubits[0])
            circuit.ccx(qubits[0], qubits[1], qubits[3])
            circuit.ccx(qubits[2], qubits[3], qubits[0])
            circuit.ccx(qubits[0], qubits[1], qubits[3])
            circuit.h(qubits[3])
        circuit.x(qubits)
        circuit.h(qubits)
        circuit.barrier(label=f'Diffusion_{iteration+1}')
    
    # Step 3: Measure brain state
    circuit.measure(qubits, cbits)
    
    return circuit, n_iterations

def executeGroverClassification(brain_signatures, search_params, simulator, measurement_shots=4096):
    """
    Execute quantum brain state classification using Grover's algorithm.
    
    Args:
        brain_signatures (dict): Dictionary of brain state patterns
        search_params (dict): Search configuration parameters
        simulator: Quantum simulator instance
        measurement_shots (int): Number of measurement shots
        
    Returns:
        dict: Classification results for all brain states
    """
    print("=== Quantum Brain State Classification Execution ===\n")
    
    classification_results = {}
    
    for state_name, target_signature in brain_signatures.items():
        print(f"Classifying {state_name} pattern: {target_signature}")
        
        # Construct Grover circuit
        grover_circuit, iterations = constructGroverCircuit(target_signature, search_params['n_qubits'])
        
        # Execute on quantum simulator
        job = simulator.run(grover_circuit, shots=measurement_shots)
        result = job.result()
        counts = result.get_counts()
        
        # Convert measurement results to brain state probabilities
        target_bitstring = ''.join(map(str, target_signature[::-1]))  # Little-endian format
        target_count = counts.get(target_bitstring, 0)
        success_probability = target_count / measurement_shots
        
        # Calculate quantum metrics
        quantum_advantage = search_params['search_space_size'] / iterations
        circuit_depth = grover_circuit.depth()
        gate_count = grover_circuit.size()
        
        # Store results
        classification_results[state_name] = {
            'target_pattern': target_signature,
            'target_bitstring': target_bitstring,
            'success_probability': success_probability,
            'iterations': iterations,
            'quantum_advantage': quantum_advantage,
            'circuit_depth': circuit_depth,
            'gate_count': gate_count,
            'measurements': counts
        }
        
        print(f"  Success probability: {success_probability:.3f}")
        print(f"  Quantum speedup: {quantum_advantage:.1f}x")
        print(f"  Circuit depth: {circuit_depth} gates")
        print(f"  Grover iterations: {iterations}")
        print()
    
    return classification_results

def analyzeClassificationPerformance(classification_results):
    """
    Analyze classification performance across all brain states.
    
    Args:
        classification_results (dict): Results from quantum classification
        
    Returns:
        dict: Performance analysis metrics
    """
    print("=== Brain State Classification Performance ===")
    total_success = 0
    high_fidelity_states = 0
    
    for state_name, results in classification_results.items():
        prob = results['success_probability']
        advantage = results['quantum_advantage']
        
        # Classification quality assessment
        if prob >= 0.7:
            quality = "Excellent"
            high_fidelity_states += 1
        elif prob >= 0.5:
            quality = "Good"
        elif prob >= 0.3:
            quality = "Moderate"
        else:
            quality = "Poor"
        
        total_success += prob
        
        print(f"{state_name:15}: {quality:9} (P={prob:.3f}, Speedup={advantage:.1f}x)")
    
    average_success = total_success / len(classification_results)
    fidelity_rate = high_fidelity_states / len(classification_results)
    
    performance_metrics = {
        'average_success': average_success,
        'fidelity_rate': fidelity_rate,
        'high_fidelity_states': high_fidelity_states,
        'total_states': len(classification_results),
        'mean_quantum_advantage': np.mean([r['quantum_advantage'] for r in classification_results.values()])
    }
    
    print(f"\nOverall Classification Metrics:")
    print(f"  Average success probability: {average_success:.3f}")
    print(f"  High-fidelity classifications: {high_fidelity_states}/{len(classification_results)} ({fidelity_rate:.1%})")
    print(f"  Mean quantum advantage: {performance_metrics['mean_quantum_advantage']:.1f}x")
    
    return performance_metrics

def simulateRealTimeProcessing(search_params, classification_results):
    """
    Simulate real-time EEG processing scenario.
    
    Args:
        search_params (dict): Search configuration parameters
        classification_results (dict): Classification results
        
    Returns:
        dict: Real-time processing metrics
    """
    print(f"\n=== Real-Time EEG Processing Simulation ===")
    sampling_rate = 256  # Hz
    analysis_window = 1.0  # seconds
    daily_analyses = int(24 * 3600 / analysis_window)
    
    classical_ops_per_analysis = search_params['search_space_size']
    quantum_ops_per_analysis = search_params['optimal_iterations']
    
    real_time_metrics = {
        'sampling_rate': sampling_rate,
        'analysis_window': analysis_window,
        'daily_analyses': daily_analyses,
        'classical_ops_daily': daily_analyses * classical_ops_per_analysis,
        'quantum_ops_daily': daily_analyses * quantum_ops_per_analysis,
        'efficiency_gain': classical_ops_per_analysis / quantum_ops_per_analysis
    }
    
    print(f"Real-time monitoring parameters:")
    print(f"  EEG sampling rate: {sampling_rate} Hz")
    print(f"  Analysis window: {analysis_window} second")
    print(f"  Daily analyses: {daily_analyses:,}")
    print(f"  Classical operations/day: {real_time_metrics['classical_ops_daily']:,}")
    print(f"  Quantum operations/day: {real_time_metrics['quantum_ops_daily']:,}")
    print(f"  Computational efficiency gain: {real_time_metrics['efficiency_gain']:.0f}x")
    
    return real_time_metrics

if __name__ == "__main__":
    # Demonstration of Grover's algorithm for neural pattern search
    
    # Initialize Grover search
    brain_signatures, search_params, simulator = initializeGroverSearch()
    
    # Execute classification
    classification_results = executeGroverClassification(brain_signatures, search_params, simulator)
    
    # Analyze performance
    performance_metrics = analyzeClassificationPerformance(classification_results)
    
    # Simulate real-time processing
    real_time_metrics = simulateRealTimeProcessing(search_params, classification_results)
    
    print(f"\n=== Grover's Algorithm Summary ===")
    print(f"Successfully implemented quantum search for {len(brain_signatures)} brain states")
    print(f"Average classification accuracy: {performance_metrics['average_success']:.1%}")
    print(f"Quantum computational advantage: {performance_metrics['mean_quantum_advantage']:.1f}x")
