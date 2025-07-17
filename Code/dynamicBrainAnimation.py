"""
Dynamic Brain Network Animation

This module creates animated visualizations of dynamic brain network activity
showing temporal evolution of connectivity patterns and network states.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
try:
    from IPython.display import Image, display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

def generateDynamicConnectivity(base_edges, atlasinfo, t):
    """
    Generate time-varying connectivity matrix with network-specific oscillations.
    
    Args:
        base_edges (numpy.ndarray): Base connectivity matrix
        atlasinfo (DataFrame): Brain atlas information
        t (float): Time point for dynamic modulation
        
    Returns:
        numpy.ndarray: Time-varying connectivity matrix
    """
    dynamic_edges = base_edges.copy()
    
    # Add oscillating components to different networks
    for i, network in enumerate(atlasinfo['yeo7networks'].unique()):
        idx = atlasinfo[atlasinfo['yeo7networks']==network].index
        
        # Each network oscillates at different frequencies (like real brain rhythms)
        freq = 0.5 + i * 0.3  # Different frequency for each network
        amplitude = 0.2       # Modulation strength
        
        # Add sinusoidal modulation to network connections
        modulation = amplitude * np.sin(freq * t)
        for row in idx:
            for col in idx:
                if row != col:
                    dynamic_edges[row, col] += modulation
                    
    return np.clip(dynamic_edges, 0, 1)  # Keep values in reasonable range

def createBrainNetworkAnimation(base_edges, atlasinfo, seqCmap, divCmap, cubehelix_reverse, save_path=None):
    """
    Create dynamic brain network activity animation.
    
    Args:
        base_edges (numpy.ndarray): Base connectivity matrix
        atlasinfo (DataFrame): Brain atlas information
        seqCmap, divCmap, cubehelix_reverse: Color palettes
        save_path (str, optional): Path to save animation
        
    Returns:
        tuple: (animation object, figure object)
    """
    # Animation parameters
    n_timepoints = 60
    time_points = np.linspace(0, 4*np.pi, n_timepoints)
    
    # Set up the animation figure with proper color scheme
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # Connectivity matrix subplot
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(base_edges, cmap=seqCmap, vmin=0, vmax=0.8)
    ax1.set_title('Dynamic Connectivity Matrix', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Network activity over time
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.set_title('Network Activity Time Series', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Network Activity Level')
    ax2.set_xlim(0, 4*np.pi)
    ax2.set_ylim(-0.3, 1.0)
    
    # Initialize activity lines for each network using project colors
    activity_lines = []
    network_names = list(atlasinfo['yeo7networks'].unique())
    colors_for_networks = seqCmap(np.linspace(0.2, 0.9, len(network_names)))
    
    for i, (network, color) in enumerate(zip(network_names, colors_for_networks)):
        line, = ax2.plot([], [], color=color, linewidth=2.5, label=network, alpha=0.8)
        activity_lines.append(line)
    
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    # Current time indicator
    time_line = ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    # Network strength bar chart
    ax3 = fig.add_subplot(gs[1, :2])
    bars = ax3.bar(network_names, [0]*len(network_names), 
                   color=colors_for_networks, alpha=0.7, edgecolor='black', linewidth=1)
    ax3.set_title('Current Network Activity Levels', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Activity Level (Normalized)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # Brain state indicator
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    state_text = ax4.text(0.5, 0.7, 'Initializing...', fontsize=12, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor=cubehelix_reverse(0.3), alpha=0.8))
    
    def animate(frame):
        """Animation function for updating plots at each frame."""
        t = time_points[frame]
        
        # Generate current connectivity matrix
        current_edges = generateDynamicConnectivity(base_edges, atlasinfo, t)
        
        # Update connectivity matrix
        im1.set_array(current_edges)
        
        # Calculate network activities (mean within-network connectivity)
        network_activities = []
        for network in network_names:
            idx = atlasinfo[atlasinfo['yeo7networks']==network].index
            if len(idx) > 1:
                # Calculate mean within-network connectivity
                network_submatrix = current_edges[np.ix_(idx, idx)]
                activity = np.mean(network_submatrix[network_submatrix > 0])
            else:
                activity = 0.5
            network_activities.append(min(activity, 1.0))
        
        # Update activity time series
        current_time = t
        for i, (line, activity) in enumerate(zip(activity_lines, network_activities)):
            x_data = list(line.get_xdata())
            y_data = list(line.get_ydata())
            
            x_data.append(current_time)
            y_data.append(activity)
            
            if len(x_data) > 100:
                x_data = x_data[-100:]
                y_data = y_data[-100:]
            
            line.set_data(x_data, y_data)
        
        # Update time indicator
        time_line.set_xdata([current_time, current_time])
        
        # Update bar chart
        for bar, activity in zip(bars, network_activities):
            bar.set_height(activity)
        
        # Update brain state based on network activity patterns
        avg_activity = np.mean(network_activities)
        dominant_network = network_names[np.argmax(network_activities)]
        
        if avg_activity > 0.7:
            state = f"High Activity\n{dominant_network} Dominant"
            color = divCmap(0.8)
        elif avg_activity > 0.5:
            state = f"Moderate Activity\n{dominant_network} Leading"
            color = seqCmap(0.6)
        else:
            state = f"Low Activity\nDistributed Processing"
            color = cubehelix_reverse(0.3)
        
        state_text.set_text(f"{state}\nTime: {current_time:.1f}s")
        state_text.get_bbox_patch().set_facecolor(color)
        
        return [im1] + activity_lines + [time_line] + list(bars) + [state_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_timepoints, 
                                  interval=150, blit=False, repeat=True)
    
    plt.suptitle('Dynamic Brain Network Activity Over Time', fontsize=16, fontweight='bold')
    
    # Save animation if path provided
    if save_path:
        try:
            anim.save(save_path, writer='pillow', fps=8, dpi=100)
            print(f"Animation saved to: {save_path}")
        except Exception as e:
            print(f"Error saving animation: {e}")
    
    return anim, fig

def createStaticDynamicsVisualization(base_edges, atlasinfo, seqCmap, save_path=None):
    """
    Create static multi-frame view of brain network dynamics as fallback.
    
    Args:
        base_edges (numpy.ndarray): Base connectivity matrix
        atlasinfo (DataFrame): Brain atlas information
        seqCmap: Color palette for visualization
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Generated static figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Brain Network Dynamics (Static Multi-Frame View)', fontsize=16, fontweight='bold')
    
    time_points = np.linspace(0, 4*np.pi, 6)
    
    for i, t in enumerate(time_points):
        ax = axes[i//3, i%3]
        dynamic_edges = generateDynamicConnectivity(base_edges, atlasinfo, t)
        im = ax.imshow(dynamic_edges[:20, :20], cmap=seqCmap)
        ax.set_title(f'Time: {t:.1f}s')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

if __name__ == "__main__":
    # Demonstration of dynamic brain network animation
    from brainNetworkSetup import createBrainAtlas, initializeVisualizationSettings
    from brainConnectivityGeneration import generateBrainConnectivity
    
    # Setup components
    seqCmap, divCmap, cubehelix_reverse = initializeVisualizationSettings()
    atlasinfo, coords, nodes_df_coords = createBrainAtlas()
    edges, network_stats = generateBrainConnectivity(atlasinfo)
    
    # Create animation
    try:
        anim, fig = createBrainNetworkAnimation(edges, atlasinfo, seqCmap, divCmap, cubehelix_reverse, 
                                               '../Plots/brain_network_animation_demo.gif')
        plt.show()
        
        # Display animation if in Jupyter
        if IPYTHON_AVAILABLE:
            display(Image('../Plots/brain_network_animation_demo.gif'))
            
    except Exception as e:
        print(f"Animation creation failed: {e}")
        print("Creating static visualization instead...")
        fig = createStaticDynamicsVisualization(edges, atlasinfo, seqCmap, 
                                               '../Plots/brain_network_static_demo.png')
        plt.show()
    
    print("\nDynamic brain network visualization complete!")
