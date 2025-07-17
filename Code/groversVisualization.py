"""
Grover's Algorithm Visualization

This module provides comprehensive visualization capabilities for Grover's
algorithm results including performance metrics and measurement analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def visualizeGroversResults(classification_results, seqCmap, divCmap, cubehelix_reverse, save_path=None):
    """
    Create comprehensive visualization of Grover's Algorithm results.
    
    Args:
        classification_results (dict): Results from Grover classification
        seqCmap, divCmap, cubehelix_reverse: Color palettes
        save_path (str, optional): Path to save the visualization
        
    Returns:
        matplotlib.figure.Figure: Generated visualization figure
    """
    # Extract data for visualization
    state_names = list(classification_results.keys())
    success_probs = [classification_results[name]['success_probability'] for name in state_names]
    quantum_complexity = [classification_results[name]['iterations'] for name in state_names]
    circuit_depths = [classification_results[name]['circuit_depth'] for name in state_names]
    gate_counts = [classification_results[name]['gate_count'] for name in state_names]
    quantum_advantages = [classification_results[name]['quantum_advantage'] for name in state_names]
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 14))
    gs = plt.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # 1. Brain State Classification Success Rates
    ax1 = fig.add_subplot(gs[0, 0])
    colors = [seqCmap(0.3 + 0.15*i) for i in range(len(state_names))]
    bars1 = ax1.bar(range(len(state_names)), success_probs, color=colors, alpha=0.8)
    
    ax1.set_title('Quantum Brain State\nClassification Success Rates', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Detection Probability')
    ax1.set_xticks(range(len(state_names)))
    ax1.set_xticklabels([name.replace('_', '\n') for name in state_names], fontsize=9)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add probability values on bars
    for bar, prob in zip(bars1, success_probs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{prob:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Quantum vs Classical Complexity Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    search_space_size = 16  # 2^4 for 4-qubit system
    classical_complexity = [search_space_size] * len(state_names)
    
    x_pos = np.arange(len(state_names))
    width = 0.35
    
    bars2a = ax2.bar(x_pos - width/2, classical_complexity, width, 
                     label='Classical O(N)', color=divCmap(0.4), alpha=0.7)
    bars2b = ax2.bar(x_pos + width/2, quantum_complexity, width,
                     label='Quantum O(√N)', color=cubehelix_reverse(0.6), alpha=0.7)
    
    ax2.set_title('Computational Complexity\nClassical vs Quantum', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Operations Required')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([name.replace('_', '\n') for name in state_names], fontsize=9)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Quantum Circuit Metrics
    ax3 = fig.add_subplot(gs[0, 2])
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(range(len(state_names)), circuit_depths, 'o-', 
                     color=seqCmap(0.7), linewidth=2, markersize=8, label='Circuit Depth')
    line2 = ax3_twin.plot(range(len(state_names)), gate_counts, 's-', 
                          color=divCmap(0.6), linewidth=2, markersize=8, label='Gate Count')
    
    ax3.set_title('Quantum Circuit\nComplexity Metrics', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Brain State')
    ax3.set_ylabel('Circuit Depth', color=seqCmap(0.7))
    ax3_twin.set_ylabel('Gate Count', color=divCmap(0.6))
    ax3.set_xticks(range(len(state_names)))
    ax3.set_xticklabels([name.replace('_', '\n') for name in state_names], fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Detailed Measurement Results for Best Performing State
    best_state = max(state_names, key=lambda x: classification_results[x]['success_probability'])
    best_results = classification_results[best_state]
    measurements = best_results['measurements']
    
    ax4 = fig.add_subplot(gs[1, :])
    
    # Convert bitstrings to readable brain state patterns
    pattern_labels = []
    pattern_counts = []
    pattern_colors = []
    
    for bitstring, count in measurements.items():
        # Convert to brain state pattern
        pattern = [int(bit) for bit in bitstring[::-1]]  # Reverse for correct order
        pattern_str = ''.join(map(str, pattern))
        pattern_labels.append(pattern_str)
        pattern_counts.append(count)
        
        # Color target pattern differently
        if pattern_str == ''.join(map(str, best_results['target_pattern'])):
            pattern_colors.append(cubehelix_reverse(0.8))  # Highlight target
        else:
            pattern_colors.append(seqCmap(0.5))
    
    # Sort by count for better visualization
    sorted_data = sorted(zip(pattern_labels, pattern_counts, pattern_colors), 
                        key=lambda x: x[1], reverse=True)
    if sorted_data:
        sorted_labels, sorted_counts, sorted_colors = zip(*sorted_data)
        
        # Show top 10 most frequent patterns
        display_n = min(10, len(sorted_labels))
        display_labels = sorted_labels[:display_n]
        display_counts = sorted_counts[:display_n]
        display_colors = sorted_colors[:display_n]
        
        bars4 = ax4.bar(range(display_n), display_counts, color=display_colors, alpha=0.8)
        
        ax4.set_title(f'Measurement Results for {best_state.replace("_", " ").title()}\n' +
                     f'Target Pattern: {best_results["target_pattern"]} (Success Rate: {best_results["success_probability"]:.1%})', 
                     fontsize=14, fontweight='bold')
        ax4.set_xlabel('Brain State Patterns [activation, left, right, motor]')
        ax4.set_ylabel('Measurement Count (out of 4096 shots)')
        ax4.set_xticks(range(display_n))
        ax4.set_xticklabels(display_labels, fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # Annotate target pattern
        target_pattern_str = ''.join(map(str, best_results['target_pattern']))
        for i, label in enumerate(display_labels):
            if label == target_pattern_str:
                ax4.annotate('TARGET\nPATTERN', xy=(i, display_counts[i]), 
                            xytext=(i, display_counts[i] + 200),
                            arrowprops=dict(arrowstyle='->', color='red', lw=2),
                            fontsize=11, fontweight='bold', ha='center', color='red')
    
    # 5. Real-Time Performance Projection
    ax5 = fig.add_subplot(gs[2, 0])
    time_windows = np.array([0.1, 0.5, 1.0, 2.0, 5.0])  # seconds
    analyses_per_day = 24 * 3600 / time_windows
    
    classical_ops = analyses_per_day * search_space_size
    quantum_ops = analyses_per_day * np.mean(quantum_complexity)
    
    ax5.semilogy(time_windows, classical_ops, 'o-', linewidth=3, markersize=8,
                 color=divCmap(0.6), label='Classical')
    ax5.semilogy(time_windows, quantum_ops, 's-', linewidth=3, markersize=8,
                 color=seqCmap(0.7), label='Quantum')
    
    ax5.set_title('Daily Operations vs\nAnalysis Window', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Analysis Window (seconds)')
    ax5.set_ylabel('Operations per Day')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Brain State Feature Encoding
    ax6 = fig.add_subplot(gs[2, 1])
    feature_names = ['Activation\nLevel', 'Left\nHemisphere', 'Right\nHemisphere', 'Motor\nCortex']
    state_patterns = np.array([classification_results[name]['target_pattern'] for name in state_names])
    
    im = ax6.imshow(state_patterns.T, cmap=seqCmap, aspect='auto', vmin=0, vmax=1)
    ax6.set_title('Brain State Feature\nEncoding Matrix', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Brain States')
    ax6.set_ylabel('Neural Features')
    ax6.set_xticks(range(len(state_names)))
    ax6.set_xticklabels([name.replace('_', '\n') for name in state_names], fontsize=9)
    ax6.set_yticks(range(len(feature_names)))
    ax6.set_yticklabels(feature_names, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    cbar.set_label('Feature Value', rotation=270, labelpad=15)
    
    # 7. Quantum Advantage Summary
    ax7 = fig.add_subplot(gs[2, 2])
    wedges, texts, autotexts = ax7.pie(quantum_advantages, labels=[name.replace('_', '\n') for name in state_names],
                                      colors=colors, autopct='%1.1fx', startangle=90)
    ax7.set_title('Quantum Speedup\nDistribution', fontsize=12, fontweight='bold')
    
    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    plt.suptitle('Grover\'s Algorithm for Quantum Brain State Classification\n' +
                f'Average Success Rate: {np.mean(success_probs):.1%} | ' +
                f'Mean Quantum Advantage: {np.mean(quantum_advantages):.1f}x',
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save the comprehensive visualization
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig

def createGroverPerformanceSummary(classification_results, performance_metrics):
    """
    Create a summary report of Grover's algorithm performance.
    
    Args:
        classification_results (dict): Classification results
        performance_metrics (dict): Performance analysis metrics
        
    Returns:
        str: Formatted performance summary
    """
    summary = "=== Grover's Algorithm Performance Summary ===\n\n"
    
    summary += f"Overall Performance:\n"
    summary += f"  • Average Success Rate: {performance_metrics['average_success']:.1%}\n"
    summary += f"  • High-Fidelity States: {performance_metrics['high_fidelity_states']}/{performance_metrics['total_states']}\n"
    summary += f"  • Mean Quantum Advantage: {performance_metrics['mean_quantum_advantage']:.1f}x\n\n"
    
    summary += f"Individual Brain State Performance:\n"
    for state_name, results in classification_results.items():
        summary += f"  • {state_name.replace('_', ' ').title()}:\n"
        summary += f"    - Success Probability: {results['success_probability']:.3f}\n"
        summary += f"    - Quantum Speedup: {results['quantum_advantage']:.1f}x\n"
        summary += f"    - Circuit Depth: {results['circuit_depth']} gates\n\n"
    
    summary += f"Technical Metrics:\n"
    total_gates = sum([r['gate_count'] for r in classification_results.values()])
    avg_depth = np.mean([r['circuit_depth'] for r in classification_results.values()])
    summary += f"  • Total Gates Used: {total_gates}\n"
    summary += f"  • Average Circuit Depth: {avg_depth:.1f}\n"
    summary += f"  • Search Space Size: 16 states (4 qubits)\n"
    summary += f"  • Optimal Iterations: {np.mean([r['iterations'] for r in classification_results.values()]):.1f}\n"
    
    return summary

if __name__ == "__main__":
    # Demonstration of Grover's algorithm visualization
    from groversNeuralSearch import initializeGroverSearch, executeGroverClassification, analyzeClassificationPerformance
    from brainNetworkSetup import initializeVisualizationSettings
    
    # Setup components
    seqCmap, divCmap, cubehelix_reverse = initializeVisualizationSettings()
    brain_signatures, search_params, simulator = initializeGroverSearch()
    
    # Execute classification
    classification_results = executeGroverClassification(brain_signatures, search_params, simulator)
    performance_metrics = analyzeClassificationPerformance(classification_results)
    
    # Create visualization
    fig = visualizeGroversResults(classification_results, seqCmap, divCmap, cubehelix_reverse,
                                 '../Plots/grovers_visualization_demo.png')
    plt.show()
    
    # Print performance summary
    summary = createGroverPerformanceSummary(classification_results, performance_metrics)
    print(summary)
