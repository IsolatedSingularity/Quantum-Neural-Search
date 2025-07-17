"""
3D Brain Network Visualization

This module provides functionality for creating 3D brain network visualizations
using the NetPlotBrain library with multiple viewing perspectives.
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    import netplotbrain
    NETPLOTBRAIN_AVAILABLE = True
except ImportError:
    NETPLOTBRAIN_AVAILABLE = False
    print("Warning: NetPlotBrain not available. 3D visualizations will be skipped.")

def create3dBrainVisualization(nodes_df_coords, atlasinfo, edges, save_prefix='../Plots/brain_networks'):
    """
    Create comprehensive 3D brain network visualizations with multiple perspectives.
    
    Args:
        nodes_df_coords (DataFrame): Node coordinates for 3D plotting
        atlasinfo (DataFrame): Atlas information with network assignments
        edges (numpy.ndarray): Brain connectivity matrix
        save_prefix (str): Prefix for saved visualization files
        
    Returns:
        list: Paths to saved visualization files
    """
    if not NETPLOTBRAIN_AVAILABLE:
        print("NetPlotBrain not available. Skipping 3D visualizations.")
        return []
    
    saved_files = []
    
    # Plot 1: Glass brain lateral view with network-colored nodes
    try:
        netplotbrain.plot(
            template='MNI152NLin2009cAsym',  # Standard brain template
            nodes=nodes_df_coords,          # Use our generated coordinates
            nodes_df=atlasinfo,             # Node information and network assignments
            edges=edges,                     # Enhanced connectivity matrix
            view='L',                       # Lateral (side) view
            template_style='glass',         # Transparent brain rendering
            node_scale=35,                  # Larger nodes for better visibility
            edge_threshold=0.3,             # Show strong connections only
            edge_thresholddirection='>',    # Threshold direction
            edge_alpha=0.4,                 # Higher edge transparency
            node_alpha=0.9,                 # High node visibility
            title='Brain Networks - Lateral View\n(Functional Connectivity with Network Structure)',
            savename=f'{save_prefix}_lateral_notebook.png'
        )
        saved_files.append(f'{save_prefix}_lateral_notebook.png')
        print("✓ Lateral view generated successfully")
    except Exception as e:
        print(f"Note: Lateral view generation had an issue: {e}")
    
    # Plot 2: Sagittal view for midline structures
    try:
        netplotbrain.plot(
            template='MNI152NLin2009cAsym',
            nodes=nodes_df_coords,
            nodes_df=atlasinfo,
            edges=edges,
            view='S',                       # Sagittal (side) view
            template_style='glass',
            node_scale=40,
            edge_threshold=0.35,            # Slightly higher threshold for clarity
            edge_thresholddirection='>',
            edge_alpha=0.3,
            title='Brain Networks - Sagittal View\n(Midline and Deep Structures)',
            savename=f'{save_prefix}_sagittal_notebook.png'
        )
        saved_files.append(f'{save_prefix}_sagittal_notebook.png')
        print("✓ Sagittal view generated successfully")
    except Exception as e:
        print(f"Note: Sagittal view generation had an issue: {e}")
    
    # Plot 3: Surface rendering for detailed anatomy
    try:
        netplotbrain.plot(
            template='MNI152NLin2009cAsym',
            nodes=nodes_df_coords,
            nodes_df=atlasinfo,
            edges=edges,
            view='L',
            template_style='surface',       # 3D surface rendering
            node_scale=45,                  # Larger nodes for surface view
            edge_threshold=0.5,             # Show strongest connections only
            edge_thresholddirection='>',
            edge_alpha=0.5,
            title='Surface-Rendered Brain Networks\n(Anatomical Detail with Network Connectivity)',
            savename=f'{save_prefix}_surface_notebook.png'
        )
        saved_files.append(f'{save_prefix}_surface_notebook.png')
        print("✓ Surface rendering generated successfully")
    except Exception as e:
        print(f"Note: Surface rendering had an issue: {e}")
    
    # Plot 4: Multiple views for comprehensive visualization
    try:
        netplotbrain.plot(
            template='MNI152NLin2009cAsym',
            nodes=nodes_df_coords,
            nodes_df=atlasinfo,
            edges=edges,
            view='LSR',                     # Left, Superior, Right views
            template_style='glass',
            node_scale=30,
            edge_threshold=0.4,             # Higher threshold for multi-view clarity
            edge_thresholddirection='>',
            edge_alpha=0.25,
            title='Brain Networks - Multiple Perspectives\n(Left, Superior, Right)',
            savename=f'{save_prefix}_multiview_notebook.png'
        )
        saved_files.append(f'{save_prefix}_multiview_notebook.png')
        print("✓ Multiple view generated successfully")
    except Exception as e:
        print(f"Note: Multiple view generation had an issue: {e}")
    
    return saved_files

def createFallback3dVisualization(nodes_df_coords, atlasinfo, edges, seqCmap, save_path=None):
    """
    Create fallback 3D-style visualization when NetPlotBrain is not available.
    
    Args:
        nodes_df_coords (DataFrame): Node coordinates
        atlasinfo (DataFrame): Atlas information
        edges (numpy.ndarray): Connectivity matrix
        seqCmap: Color map for visualization
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Generated fallback figure
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Create 2x2 subplot layout for different "views"
    views = ['Lateral View', 'Sagittal View', 'Superior View', 'Connectivity Detail']
    
    for i, view_name in enumerate(views):
        ax = fig.add_subplot(2, 2, i+1, projection='3d' if i < 3 else None)
        
        if i < 3:  # 3D scatter plots
            # Color nodes by network
            networks = atlasinfo['yeo7networks'].unique()
            colors = seqCmap(np.linspace(0, 1, len(networks)))
            
            for j, network in enumerate(networks):
                mask = atlasinfo['yeo7networks'] == network
                ax.scatter(nodes_df_coords.loc[mask, 'x'], 
                          nodes_df_coords.loc[mask, 'y'],
                          nodes_df_coords.loc[mask, 'z'],
                          c=[colors[j]], label=network, s=60, alpha=0.7)
            
            ax.set_title(f'Brain Networks - {view_name}')
            ax.legend(fontsize=8)
        else:  # Connectivity matrix
            im = ax.imshow(edges, cmap=seqCmap, aspect='auto')
            ax.set_title('Connectivity Matrix')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Brain Network Visualization (Fallback Mode)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig

if __name__ == "__main__":
    # Demonstration of 3D brain visualization
    from brainNetworkSetup import createBrainAtlas, initializeVisualizationSettings
    from brainConnectivityGeneration import generateBrainConnectivity, enhanceNetworkConnectivity
    
    # Setup components
    seqCmap, divCmap, cubehelix_reverse = initializeVisualizationSettings()
    atlasinfo, coords, nodes_df_coords = createBrainAtlas()
    edges, network_stats = generateBrainConnectivity(atlasinfo)
    enhanced_edges = enhanceNetworkConnectivity(edges, atlasinfo)
    
    # Create 3D visualizations
    if NETPLOTBRAIN_AVAILABLE:
        saved_files = create3dBrainVisualization(nodes_df_coords, atlasinfo, enhanced_edges)
        print(f"\n3D visualizations complete! Saved {len(saved_files)} files.")
    else:
        fig = createFallback3dVisualization(nodes_df_coords, atlasinfo, enhanced_edges, seqCmap, 
                                          '../Plots/brain_3d_fallback.png')
        plt.show()
        print("\nFallback 3D visualization complete!")
