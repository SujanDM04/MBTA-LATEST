import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def plot_cost_progress(sa_cost_progress, hc_cost_progress, save_dir="plots"):
    """
    Plot cost function progress for both Simulated Annealing and Hill Climbing algorithms.
    
    Args:
        sa_cost_progress: List of costs from Simulated Annealing optimization
        hc_cost_progress: List of costs from Hill Climbing optimization
        save_dir: Directory to save the plots
    """
    # Create plots directory if it doesn't exist
    Path(save_dir).mkdir(exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Simulated Annealing Cost Progress
    iterations_sa = range(len(sa_cost_progress))
    ax1.plot(iterations_sa, sa_cost_progress, 'b-', linewidth=1.5, label='Current Cost')
    ax1.set_title('Simulated Annealing - Cost Function Progress', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Total Cost', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add final cost annotation for SA
    final_cost_sa = sa_cost_progress[-1] if sa_cost_progress else 0
    ax1.annotate(f'Final Cost: {final_cost_sa:,.0f}', 
                xy=(len(sa_cost_progress)-1, final_cost_sa),
                xytext=(len(sa_cost_progress)*0.7, final_cost_sa*1.2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    # Plot 2: Hill Climbing Cost Progress
    iterations_hc = range(len(hc_cost_progress))
    ax2.plot(iterations_hc, hc_cost_progress, 'g-', linewidth=1.5, label='Current Cost')
    ax2.set_title('Hill Climbing - Cost Function Progress', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Total Cost', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add final cost annotation for HC
    final_cost_hc = hc_cost_progress[-1] if hc_cost_progress else 0
    ax2.annotate(f'Final Cost: {final_cost_hc:,.0f}', 
                xy=(len(hc_cost_progress)-1, final_cost_hc),
                xytext=(len(hc_cost_progress)*0.7, final_cost_hc*1.2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cost_progress_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also create individual plots
    # Individual SA plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations_sa, sa_cost_progress, 'b-', linewidth=2)
    plt.title('Simulated Annealing - Cost Function Progress', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Total Cost', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/sa_cost_progress.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual HC plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations_hc, hc_cost_progress, 'g-', linewidth=2)
    plt.title('Hill Climbing - Cost Function Progress', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Total Cost', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/hc_cost_progress.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cost progress plots saved in '{save_dir}' directory:")
    print(f"- {save_dir}/cost_progress_comparison.png (combined view)")
    print(f"- {save_dir}/sa_cost_progress.png (Simulated Annealing only)")
    print(f"- {save_dir}/hc_cost_progress.png (Hill Climbing only)")

def main():
    """
    Main function to run the cost progress plotting.
    This function should be called after running the optimization algorithms.
    """
    print("Cost Progress Plotting Script")
    print("=" * 40)
    
    # Example usage - you would typically get these from your main.py execution
    # For now, we'll create some sample data to demonstrate the plotting
    print("Note: This script is designed to be called after running main.py")
    print("To use with actual optimization results, modify main.py to call this function")
    print("with the actual cost_progress lists from your optimization runs.")
    
    # Sample data for demonstration
    sa_sample = [10000, 9500, 9200, 9000, 8800, 8500, 8200, 8000, 7800, 7500]
    hc_sample = [10000, 9800, 9600, 9400, 9200, 9000, 8800, 8600, 8400, 8200]
    
    print("\nPlotting sample cost progress data...")
    plot_cost_progress(sa_sample, hc_sample)

if __name__ == "__main__":
    main() 