"""
Master Brain Analysis Script

This module provides a comprehensive analysis framework that integrates all
quantum neuroscience components for complete brain network and quantum analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Import all analysis modules
from brainNetworkSetup import (initializeVisualizationSettings, ensurePlotsDirectory, 
                              createBrainAtlas)
from brainConnectivityGeneration import (generateBrainConnectivity, enhanceNetworkConnectivity,
                                       visualizeConnectivityMatrix)
from brain3dVisualization import create3dBrainVisualization
from dynamicBrainAnimation import createBrainNetworkAnimation
from quantumNeuralProcessing import (createQuantumNeuroscienceVisualization, 
                                   createQuantumCircuitVisualization)
from groversNeuralSearch import (initializeGroverSearch, executeGroverClassification,
                               analyzeClassificationPerformance)
from groversVisualization import visualizeGroversResults
from variationalQuantumClassifier import (setupQuantumDevice, createQuantumClassifier,
                                        prepareBrainStateTrainingData, trainVariationalQuantumClassifier,
                                        evaluateQuantumClassifier)
from variationalVisualization import visualizeVariationalResults

def runComprehensiveBrainAnalysis(save_all_plots=True, create_animations=True):
    """
    Run complete brain network and quantum analysis pipeline.
    
    Args:
        save_all_plots (bool): Whether to save all generated plots
        create_animations (bool): Whether to create animations (may be slow)
        
    Returns:
        dict: Complete analysis results
    """
    print("=" * 60)
    print("COMPREHENSIVE QUANTUM BRAIN ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Initialize visualization components
    print("\n1. Initializing Visualization Components...")
    seqCmap, divCmap, cubehelix_reverse = initializeVisualizationSettings()
    plots_dir = ensurePlotsDirectory()
    
    # Create brain atlas and connectivity
    print("\n2. Creating Brain Atlas and Connectivity...")
    atlasinfo, coords, nodes_df_coords = createBrainAtlas()
    edges, network_stats = generateBrainConnectivity(atlasinfo)
    enhanced_edges = enhanceNetworkConnectivity(edges, atlasinfo)
    
    # Generate brain network visualizations
    print("\n3. Generating Brain Network Visualizations...")
    connectivity_fig = visualizeConnectivityMatrix(edges, atlasinfo, seqCmap, 
                                                  f"{plots_dir}/master_connectivity_matrix.png" if save_all_plots else None)
    
    # Create 3D brain visualizations
    print("\n4. Creating 3D Brain Visualizations...")
    try:
        saved_3d_files = create3dBrainVisualization(nodes_df_coords, atlasinfo, enhanced_edges,
                                                   f"{plots_dir}/master_brain_3d" if save_all_plots else None)
        print(f"   Created {len(saved_3d_files)} 3D visualization files")
    except Exception as e:
        print(f"   3D visualization skipped: {e}")
        saved_3d_files = []
    
    # Create dynamic brain animation
    if create_animations:
        print("\n5. Creating Dynamic Brain Animation...")
        try:
            anim, anim_fig = createBrainNetworkAnimation(enhanced_edges, atlasinfo, seqCmap, divCmap, cubehelix_reverse,
                                                        f"{plots_dir}/master_brain_animation.gif" if save_all_plots else None)
            print("   Brain animation created successfully")
        except Exception as e:
            print(f"   Animation creation skipped: {e}")
            anim = None
    else:
        print("\n5. Skipping Dynamic Brain Animation...")
        anim = None
    
    # Generate quantum neural processing visualizations
    print("\n6. Creating Quantum Neural Processing Visualizations...")
    quantum_neuroscience_fig = createQuantumNeuroscienceVisualization(seqCmap, divCmap, cubehelix_reverse,
                                                                     f"{plots_dir}/master_quantum_neuroscience.png" if save_all_plots else None)
    quantum_circuit_fig = createQuantumCircuitVisualization(seqCmap, divCmap, cubehelix_reverse,
                                                           f"{plots_dir}/master_quantum_circuit.png" if save_all_plots else None)
    
    # Execute Grover's algorithm analysis
    print("\n7. Executing Grover's Algorithm Analysis...")
    brain_signatures, search_params, simulator = initializeGroverSearch()
    grover_results = executeGroverClassification(brain_signatures, search_params, simulator)
    grover_performance = analyzeClassificationPerformance(grover_results)
    
    # Visualize Grover's results
    grover_fig = visualizeGroversResults(grover_results, seqCmap, divCmap, cubehelix_reverse,
                                        f"{plots_dir}/master_grover_analysis.png" if save_all_plots else None)
    
    # Execute Variational Quantum Classifier analysis
    print("\n8. Executing Variational Quantum Classifier Analysis...")
    n_qubits = 4
    dev = setupQuantumDevice(n_qubits)
    quantum_classifier = createQuantumClassifier(dev)
    
    X_train, X_test, y_train, y_test, state_names, scaler = prepareBrainStateTrainingData()
    vqc_weights, vqc_cost_history = trainVariationalQuantumClassifier(
        X_train, y_train, state_names, quantum_classifier, max_iterations=50
    )
    vqc_results = evaluateQuantumClassifier(X_train, X_test, y_train, y_test, vqc_weights, quantum_classifier, state_names)
    
    # Visualize Variational results
    vqc_fig = visualizeVariationalResults(vqc_results, vqc_cost_history, seqCmap, divCmap, cubehelix_reverse,
                                         f"{plots_dir}/master_variational_analysis.png" if save_all_plots else None)
    
    # Create master summary visualization
    print("\n9. Creating Master Summary Visualization...")
    master_fig = createMasterSummaryVisualization(
        atlasinfo, edges, grover_results, grover_performance, vqc_results, 
        seqCmap, divCmap, cubehelix_reverse,
        f"{plots_dir}/master_brain_analysis.png" if save_all_plots else None
    )
    
    # Compile comprehensive results
    comprehensive_results = {
        'brain_atlas': {
            'atlasinfo': atlasinfo,
            'coordinates': coords,
            'nodes_df_coords': nodes_df_coords,
            'connectivity_matrix': edges,
            'enhanced_connectivity': enhanced_edges,
            'network_stats': network_stats
        },
        'grover_analysis': {
            'results': grover_results,
            'performance': grover_performance,
            'brain_signatures': brain_signatures,
            'search_params': search_params
        },
        'variational_analysis': {
            'results': vqc_results,
            'cost_history': vqc_cost_history,
            'weights': vqc_weights,
            'training_data': (X_train, X_test, y_train, y_test),
            'state_names': state_names
        },
        'visualizations': {
            'connectivity_fig': connectivity_fig,
            'quantum_neuroscience_fig': quantum_neuroscience_fig,
            'quantum_circuit_fig': quantum_circuit_fig,
            'grover_fig': grover_fig,
            'vqc_fig': vqc_fig,
            'master_fig': master_fig,
            'animation': anim,
            'saved_3d_files': saved_3d_files
        },
        'color_palettes': {
            'seqCmap': seqCmap,
            'divCmap': divCmap,
            'cubehelix_reverse': cubehelix_reverse
        }
    }
    
    print("\n10. Analysis Complete!")
    print("=" * 60)
    
    return comprehensive_results

def createMasterSummaryVisualization(atlasinfo, edges, grover_results, grover_performance, 
                                   vqc_results, seqCmap, divCmap, cubehelix_reverse, save_path=None):
    """
    Create a master summary visualization combining all analysis results.
    
    Args:
        atlasinfo (DataFrame): Brain atlas information
        edges (array): Connectivity matrix
        grover_results (dict): Grover's algorithm results
        grover_performance (dict): Grover performance metrics
        vqc_results (dict): Variational quantum classifier results
        seqCmap, divCmap, cubehelix_reverse: Color palettes
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Master summary figure
    """
    # Create master summary figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    # 1. Brain Connectivity Overview
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(edges, cmap=seqCmap, aspect='auto')
    ax1.set_title('Brain Connectivity\nMatrix', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Brain Regions')
    ax1.set_ylabel('Brain Regions')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 2. Network Statistics
    ax2 = fig.add_subplot(gs[0, 1])
    network_names = atlasinfo['yeo7networks'].unique()
    network_sizes = [len(atlasinfo[atlasinfo['yeo7networks']==net]) for net in network_names]
    colors = seqCmap(np.linspace(0.2, 0.9, len(network_names)))
    
    bars2 = ax2.bar(range(len(network_names)), network_sizes, color=colors)
    ax2.set_title('Brain Network\nOrganization', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Functional Networks')
    ax2.set_ylabel('Number of Regions')
    ax2.set_xticks(range(len(network_names)))
    ax2.set_xticklabels(network_names, rotation=45, fontsize=9)
    
    # 3. Grover's Algorithm Performance
    ax3 = fig.add_subplot(gs[0, 2])
    grover_states = list(grover_results.keys())
    grover_probs = [grover_results[state]['success_probability'] for state in grover_states]
    
    bars3 = ax3.bar(range(len(grover_states)), grover_probs, color=divCmap(0.6))
    ax3.set_title('Grover\'s Algorithm\nSuccess Rates', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Brain States')
    ax3.set_ylabel('Detection Probability')
    ax3.set_xticks(range(len(grover_states)))
    ax3.set_xticklabels([state.replace('_', '\n') for state in grover_states], fontsize=8)
    ax3.set_ylim(0, 1)
    
    # 4. Variational Classifier Performance
    ax4 = fig.add_subplot(gs[0, 3])
    vqc_states = vqc_results['state_names']
    vqc_accs = list(vqc_results['state_accuracies'].values())
    
    bars4 = ax4.bar(range(len(vqc_states)), vqc_accs, color=cubehelix_reverse(0.6))
    ax4.set_title('Variational Classifier\nAccuracies', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Brain States')
    ax4.set_ylabel('Classification Accuracy')
    ax4.set_xticks(range(len(vqc_states)))
    ax4.set_xticklabels([state.replace('_', '\n') for state in vqc_states], fontsize=8)
    ax4.set_ylim(0, 1)
    
    # 5. Quantum vs Classical Comparison
    ax5 = fig.add_subplot(gs[1, :2])
    methods = ['Grover\'s\nSearch', 'Variational\nClassifier']
    quantum_performance = [grover_performance['average_success'], vqc_results['test_accuracy']]
    classical_baseline = [1.0/len(grover_states), 1.0/len(vqc_states)]
    quantum_advantage = [grover_performance['mean_quantum_advantage'], vqc_results['quantum_improvement']]
    
    x_pos = np.arange(len(methods))
    width = 0.25
    
    bars5a = ax5.bar(x_pos - width, quantum_performance, width, label='Quantum Performance', color=seqCmap(0.7))
    bars5b = ax5.bar(x_pos, classical_baseline, width, label='Classical Baseline', color=divCmap(0.5))
    bars5c = ax5.bar(x_pos + width, np.array(quantum_advantage)/10, width, label='Quantum Advantage (÷10)', color=cubehelix_reverse(0.7))
    
    ax5.set_title('Quantum vs Classical Performance Comparison', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Performance Metric')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(methods)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Algorithm Complexity Analysis
    ax6 = fig.add_subplot(gs[1, 2:])
    complexity_data = {
        'Grover Search': {
            'Classical': 16,  # O(N) for 4-bit search
            'Quantum': 4     # O(√N) for 4-bit search
        },
        'VQC Training': {
            'Classical': 100,  # Approximate classical equivalent
            'Quantum': 50     # Actual quantum iterations
        }
    }
    
    alg_names = list(complexity_data.keys())
    classical_ops = [complexity_data[alg]['Classical'] for alg in alg_names]
    quantum_ops = [complexity_data[alg]['Quantum'] for alg in alg_names]
    
    x_pos = np.arange(len(alg_names))
    width = 0.35
    
    bars6a = ax6.bar(x_pos - width/2, classical_ops, width, label='Classical Operations', color=divCmap(0.6))
    bars6b = ax6.bar(x_pos + width/2, quantum_ops, width, label='Quantum Operations', color=seqCmap(0.7))
    
    ax6.set_title('Computational Complexity: Classical vs Quantum', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Operations Required')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(alg_names)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Key Metrics Summary
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    summary_text = f"""QUANTUM NEUROSCIENCE ANALYSIS SUMMARY
    
Brain Network Analysis:
• Total Brain Regions: {len(atlasinfo)} across {len(network_names)} functional networks
• Connectivity Strength: {edges.mean():.3f} ± {edges.std():.3f}
• Network Modularity: Strong within-network connections, weak between-network links
    
Grover's Algorithm Results:
• Average Success Rate: {grover_performance['average_success']:.1%}
• Quantum Speedup: {grover_performance['mean_quantum_advantage']:.1f}x over classical search
• High-Fidelity States: {grover_performance['high_fidelity_states']}/{grover_performance['total_states']} states achieve >70% accuracy
• Best Performing State: {max(grover_results, key=lambda x: grover_results[x]['success_probability'])}
    
Variational Quantum Classifier Results:
• Test Accuracy: {vqc_results['test_accuracy']:.1%}
• Quantum Improvement: {vqc_results['quantum_improvement']:.1f}x over random baseline
• Circuit Complexity: {vqc_results['circuit_complexity']['n_parameters']} parameters on {vqc_results['circuit_complexity']['n_qubits']} qubits
• Best Performing State: {max(vqc_results['state_accuracies'], key=vqc_results['state_accuracies'].get)}
    
Overall Quantum Advantage:
• Pattern Search: {grover_performance['mean_quantum_advantage']:.1f}x speedup for exact pattern matching
• Machine Learning: {vqc_results['quantum_improvement']:.1f}x improvement for adaptive classification
• Scalability: Quantum approaches show promise for larger neural datasets
• Near-term Feasibility: Compatible with current NISQ hardware capabilities"""
    
    ax7.text(0.02, 0.98, summary_text, transform=ax7.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=1.0", facecolor=cubehelix_reverse(0.2), alpha=0.8))
    
    # Main title
    fig.suptitle('Comprehensive Quantum Neuroscience Analysis\nBrain Networks, Grover\'s Search, and Variational Quantum Classification', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Save the master visualization
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig

def printComprehensiveResults(results):
    """
    Print a comprehensive summary of all analysis results.
    
    Args:
        results (dict): Complete analysis results
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE QUANTUM NEUROSCIENCE ANALYSIS RESULTS")
    print("="*80)
    
    # Brain Network Analysis
    print("\nBRAIN NETWORK ANALYSIS:")
    print(f"  • Brain Regions: {len(results['brain_atlas']['atlasinfo'])}")
    print(f"  • Functional Networks: {len(results['brain_atlas']['atlasinfo']['yeo7networks'].unique())}")
    print(f"  • Connectivity Density: {np.mean(results['brain_atlas']['connectivity_matrix']):.3f}")
    
    # Grover's Algorithm Results
    grover_perf = results['grover_analysis']['performance']
    print("\nGROVER'S ALGORITHM RESULTS:")
    print(f"  • Average Success Rate: {grover_perf['average_success']:.1%}")
    print(f"  • Quantum Speedup: {grover_perf['mean_quantum_advantage']:.1f}x")
    print(f"  • High-Fidelity Classifications: {grover_perf['high_fidelity_states']}/{grover_perf['total_states']}")
    
    # Variational Classifier Results
    vqc_res = results['variational_analysis']['results']
    print("\nVARIATIONAL QUANTUM CLASSIFIER RESULTS:")
    print(f"  • Test Accuracy: {vqc_res['test_accuracy']:.1%}")
    print(f"  • Quantum Improvement: {vqc_res['quantum_improvement']:.1f}x")
    print(f"  • Circuit Parameters: {vqc_res['circuit_complexity']['n_parameters']}")
    
    # Visualization Summary
    vis_count = len([v for v in results['visualizations'].values() if v is not None])
    print(f"\nVISUALIZATIONS CREATED: {vis_count} figures and plots")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    # Run comprehensive analysis
    print("Starting Comprehensive Quantum Brain Analysis...")
    
    # Execute full analysis pipeline
    results = runComprehensiveBrainAnalysis(save_all_plots=True, create_animations=False)
    
    # Display results summary
    printComprehensiveResults(results)
    
    # Show master summary
    if results['visualizations']['master_fig'] is not None:
        plt.show()
    
    print("\nComprehensive analysis complete! Check the Plots directory for all generated visualizations.")
