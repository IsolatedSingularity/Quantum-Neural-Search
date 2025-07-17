"""
Brain Connectivity Matrix Generation

This module provides functionality for generating realistic brain connectivity
patterns that mirror the organizational principles found in real brain networks.
"""

import numpy as np
import pandas as pd
import itertools
from matplotlib import pyplot as plt
import seaborn as sns

def generateBrainConnectivity(atlasinfo, connectivity_seed=42):
    """
    Generate realistic brain connectivity matrix with network structure.
    
    Args:
        atlasinfo (DataFrame): Brain atlas information with network assignments
        connectivity_seed (int): Random seed for reproducible connectivity patterns
        
    Returns:
        tuple: (edges matrix, network_stats DataFrame)
    """
    print("Generating brain connectivity matrix...")
    
    np.random.seed(connectivity_seed)
    n_regions = len(atlasinfo)
    
    # Initialize connectivity matrix with small random baseline connections
    edges = np.random.normal(0, 0.025, [n_regions, n_regions])
    
    # Create stronger within-network connections (realistic brain organization)
    # This reflects the principle that brain regions within the same functional network
    # tend to be more strongly connected than regions across networks
    for network in atlasinfo['yeo7networks'].unique():
        # Find indices of regions belonging to this network
        network_indices = atlasinfo[atlasinfo['yeo7networks'] == network].index
        
        # Create all possible pairs within this network
        network_pairs = np.array(list(itertools.combinations(network_indices, 2)))
        
        if len(network_pairs) > 0:
            # Set stronger within-network connectivity
            within_network_strength = np.random.normal(0.5, 0.05, len(network_pairs))
            
            # Apply symmetric connectivity (brain networks are typically undirected)
            edges[network_pairs[:, 0], network_pairs[:, 1]] = within_network_strength
            edges[network_pairs[:, 1], network_pairs[:, 0]] = within_network_strength
    
    # Ensure diagonal is zero (no self-connections)
    np.fill_diagonal(edges, 0)
    
    # Display connectivity statistics
    print(f"Connectivity matrix shape: {edges.shape}")
    print(f"Connection strength range: {edges.min():.3f} to {edges.max():.3f}")
    print(f"Mean connectivity strength: {edges.mean():.3f}")
    print(f"Number of strong connections (>0.3): {np.sum(edges > 0.3)}")
    
    # Analyze network structure
    network_stats = []
    for network in atlasinfo['yeo7networks'].unique():
        network_indices = atlasinfo[atlasinfo['yeo7networks'] == network].index
        network_size = len(network_indices)
        
        # Calculate within-network connectivity
        if network_size > 1:
            within_connections = edges[np.ix_(network_indices, network_indices)]
            mean_within = np.mean(within_connections[within_connections != 0])
        else:
            mean_within = 0
        
        network_stats.append({
            'Network': network,
            'Regions': network_size,
            'Mean_Connectivity': mean_within
        })
    
    network_df = pd.DataFrame(network_stats)
    print("\nNetwork connectivity statistics:")
    print(network_df)
    
    return edges, network_df

def enhanceNetworkConnectivity(edges, atlasinfo):
    """
    Strengthen within-network connections for better visualization.
    
    Args:
        edges (numpy.ndarray): Existing connectivity matrix
        atlasinfo (DataFrame): Brain atlas information
        
    Returns:
        numpy.ndarray: Enhanced connectivity matrix
    """
    enhanced_edges = edges.copy()
    
    # Strengthen within-network connections for better visual clarity
    for network in atlasinfo['yeo7networks'].unique():
        network_indices = atlasinfo[atlasinfo['yeo7networks'] == network].index
        if len(network_indices) > 1:
            # Create strong within-network connections
            for i in network_indices:
                for j in network_indices:
                    if i != j:
                        enhanced_edges[i, j] = np.random.normal(0.7, 0.05)  # Strong positive connections
    
    return enhanced_edges

def visualizeConnectivityMatrix(edges, atlasinfo, seqCmap, save_path=None):
    """
    Create visualization of the brain connectivity matrix.
    
    Args:
        edges (numpy.ndarray): Connectivity matrix to visualize
        atlasinfo (DataFrame): Brain atlas information
        seqCmap: Sequential colormap for visualization
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Create the connectivity matrix heatmap using the approved color palette
    im = ax.imshow(edges, cmap=seqCmap, aspect='auto', vmin=0, vmax=0.6)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Connectivity Strength', fontsize=12)
    
    # Customize the plot
    ax.set_title('Brain Connectivity Matrix\n(33 Regions, 6 Functional Networks)', 
                 fontsize=16, fontweight='bold', pad=60)
    ax.set_xlabel('Brain Regions', fontsize=14)
    ax.set_ylabel('Brain Regions', fontsize=14)
    
    # Add network boundaries for visual organization
    network_boundaries = []
    current_idx = 0
    for network in atlasinfo['yeo7networks'].unique():
        network_size = len(atlasinfo[atlasinfo['yeo7networks'] == network])
        network_boundaries.append(current_idx + network_size)
        current_idx += network_size
    
    # Draw network boundary lines
    for boundary in network_boundaries[:-1]:  # Exclude the last boundary
        ax.axhline(y=boundary-0.5, color='white', linewidth=2, alpha=0.7)
        ax.axvline(x=boundary-0.5, color='white', linewidth=2, alpha=0.7)
    
    # Add network labels (simplified for clarity)
    network_centers = []
    start_idx = 0
    for i, network in enumerate(atlasinfo['yeo7networks'].unique()):
        network_size = len(atlasinfo[atlasinfo['yeo7networks'] == network])
        center = start_idx + network_size // 2
        network_centers.append((center, network))
        start_idx += network_size
    
    # Add text labels for networks
    for center, network in network_centers:
        ax.text(center, -3, network, ha='center', va='top', fontsize=10, fontweight='bold')
        ax.text(-3, center, network, ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)
    
    plt.tight_layout()
    
    # Save to specified path if provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig

if __name__ == "__main__":
    # Demonstration of connectivity matrix generation
    from brainNetworkSetup import createBrainAtlas, initializeVisualizationSettings
    
    # Setup components
    seqCmap, divCmap, cubehelix_reverse = initializeVisualizationSettings()
    atlasinfo, coords, nodes_df_coords = createBrainAtlas()
    
    # Generate connectivity
    edges, network_stats = generateBrainConnectivity(atlasinfo)
    
    # Create visualization
    fig = visualizeConnectivityMatrix(edges, atlasinfo, seqCmap, '../Plots/connectivity_demo.png')
    plt.show()
    
    print("\nConnectivity matrix generation complete!")
