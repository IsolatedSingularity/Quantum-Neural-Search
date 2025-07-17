"""
Brain Network Setup and Configuration

This module provides functionality for setting up brain network visualization
components including atlas configuration, region definitions, and color schemes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def initializeVisualizationSettings():
    """
    Initialize visualization settings and color palettes for consistent plotting.
    
    Returns:
        tuple: (seqCmap, divCmap, cubehelix_reverse) color palettes
    """
    # Set random seed for reproducibility
    np.random.seed(2022)
    
    # Define consistent color palettes as per project standards
    seqCmap = sns.color_palette("mako", as_cmap=True)
    divCmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
    cubehelix_reverse = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
    
    # Configure matplotlib for high-quality plots
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    
    print("Libraries loaded successfully!")
    print("Color palettes configured:")
    print("- Sequential: Mako")
    print("- Diverging: Cubehelix")
    print("- Light: Cubehelix Reverse")
    
    return seqCmap, divCmap, cubehelix_reverse

def ensurePlotsDirectory():
    """
    Create plots directory if it doesn't exist.
    
    Returns:
        str: Path to plots directory
    """
    plots_dir = '../Plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created plots directory: {plots_dir}")
    else:
        print(f"Plots directory already exists: {plots_dir}")
    
    return plots_dir

def createBrainAtlas():
    """
    Create brain atlas with functional network organization.
    
    Returns:
        tuple: (atlasinfo DataFrame, coordinates array, nodes_df_coords DataFrame)
    """
    print("Setting up brain atlas and connectivity data...")
    
    # Define brain regions based on major functional networks
    brain_regions = [
        # Default Mode Network
        'mPFC', 'PCC', 'Angular_L', 'Angular_R', 'ITG_L', 'ITG_R',
        # Salience Network  
        'dACC', 'AI_L', 'AI_R', 'VLPFC_L', 'VLPFC_R',
        # Central Executive Network
        'DLPFC_L', 'DLPFC_R', 'IPS_L', 'IPS_R', 'FEF_L', 'FEF_R',
        # Sensorimotor Network
        'M1_L', 'M1_R', 'S1_L', 'S1_R', 'SMA_L', 'SMA_R',
        # Visual Network
        'V1_L', 'V1_R', 'V2_L', 'V2_R', 'MT_L', 'MT_R',
        # Auditory Network
        'A1_L', 'A1_R', 'STG_L', 'STG_R'
    ]
    
    # Network assignments for functional analysis
    network_labels = ['DMN']*6 + ['SN']*5 + ['CEN']*6 + ['SMN']*6 + ['VIS']*6 + ['AUD']*4
    
    # Create atlas info dataframe
    atlasinfo = pd.DataFrame({
        'name': brain_regions,
        'network': network_labels,
        'hemisphere': ['L' if '_L' in name else 'R' if '_R' in name else 'M' for name in brain_regions]
    })
    
    # Generate realistic 3D coordinates for brain regions (MNI space approximation)
    np.random.seed(42)  # For reproducible coordinates
    n_regions = len(brain_regions)
    
    # Create coordinates that roughly follow brain anatomy
    coords = []
    for i, (name, network) in enumerate(zip(brain_regions, network_labels)):
        # Base coordinates for different networks
        if network == 'DMN':
            base = [0, -50, 30] if 'PCC' in name else [0, 50, 0]  # Posterior/anterior midline
        elif network == 'SN':
            base = [40, 20, 0] if '_R' in name else [-40, 20, 0]  # Insula regions
        elif network == 'CEN':
            base = [45, 25, 35] if '_R' in name else [-45, 25, 35]  # Frontal-parietal
        elif network == 'SMN':
            base = [40, -20, 50] if '_R' in name else [-40, -20, 50]  # Motor strip
        elif network == 'VIS':
            base = [25, -80, 0] if '_R' in name else [-25, -80, 0]  # Occipital
        else:  # AUD
            base = [50, -25, 10] if '_R' in name else [-50, -25, 10]  # Temporal
        
        # Add some variation
        coord = [base[0] + np.random.normal(0, 5), 
                 base[1] + np.random.normal(0, 5), 
                 base[2] + np.random.normal(0, 5)]
        coords.append(coord)
    
    coords = np.array(coords)
    
    # Create proper nodes DataFrame for NetPlotBrain
    nodes_df_coords = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1], 
        'z': coords[:, 2]
    })
    
    # Use the network labels we already defined
    atlasinfo['yeo7networks'] = atlasinfo['network']
    
    print(f"Atlas created: {len(atlasinfo)} brain regions across {len(atlasinfo['yeo7networks'].unique())} networks")
    
    return atlasinfo, coords, nodes_df_coords

if __name__ == "__main__":
    # Demonstration of the brain network setup
    seqCmap, divCmap, cubehelix_reverse = initializeVisualizationSettings()
    plots_dir = ensurePlotsDirectory()
    atlasinfo, coords, nodes_df_coords = createBrainAtlas()
    
    print("\nBrain Network Setup Complete!")
    print(f"Created atlas with {len(atlasinfo)} regions")
    print(f"Networks: {list(atlasinfo['yeo7networks'].unique())}")
