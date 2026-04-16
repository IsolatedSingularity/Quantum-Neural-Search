"""Generate all plots headlessly (no display required).

Calls runComprehensiveBrainAnalysis which is self-contained: it generates
brain atlas data, Grover results, VQC results, and passes them to each
visualization function (visualizeGroversResults, visualizeVariationalResults,
create3dBrainVisualization) with the correct arguments.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    from quantumNeuralSearch.visualization.masterAnalysis import runComprehensiveBrainAnalysis

    print("Running comprehensive quantum brain analysis pipeline...")
    comprehensiveResults = runComprehensiveBrainAnalysis(
        save_all_plots=True, create_animations=False
    )
    plt.close("all")

    saved3dFiles = comprehensiveResults.get("visualizations", {}).get("saved_3d_files", [])
    print(f"Done. All plots saved to Plots/ ({len(saved3dFiles)} 3D files)")


if __name__ == "__main__":
    main()
