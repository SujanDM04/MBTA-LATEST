import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def plot_cost_progress(sa_cost_progress, hc_cost_progress, ga_cost_progress=None, save_dir="plots"):
    """
    Plot cost function progress for Simulated Annealing, Hill Climbing, and optionally Genetic Algorithm on the same plot.
    Args:
        sa_cost_progress: List of costs from Simulated Annealing optimization
        hc_cost_progress: List of costs from Hill Climbing optimization
        ga_cost_progress: List of costs from Genetic Algorithm optimization (optional)
        save_dir: Directory to save the plots
    """
    # Create plots directory if it doesn't exist
    Path(save_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(12, 7))
    iterations_sa = range(len(sa_cost_progress))
    iterations_hc = range(len(hc_cost_progress))
    plt.plot(iterations_sa, sa_cost_progress, 'b-', linewidth=2, label='Simulated Annealing')
    plt.plot(iterations_hc, hc_cost_progress, 'g-', linewidth=2, label='Hill Climbing')
    if ga_cost_progress is not None:
        iterations_ga = range(len(ga_cost_progress))
        plt.plot(iterations_ga, ga_cost_progress, 'r-', linewidth=2, label='Genetic Algorithm')
        # Annotate final GA cost
        if ga_cost_progress:
            plt.annotate(f'Final GA Cost: {ga_cost_progress[-1]:,.0f}',
                         xy=(len(ga_cost_progress)-1, ga_cost_progress[-1]),
                         xytext=(len(ga_cost_progress)*0.7, ga_cost_progress[-1]*1.05),
                         arrowprops=dict(arrowstyle='->', color='red'),
                         fontsize=10, color='red')
    # Annotate final costs for SA and HC
    if sa_cost_progress:
        plt.annotate(f'Final SA Cost: {sa_cost_progress[-1]:,.0f}',
                     xy=(len(sa_cost_progress)-1, sa_cost_progress[-1]),
                     xytext=(len(sa_cost_progress)*0.7, sa_cost_progress[-1]*1.1),
                     arrowprops=dict(arrowstyle='->', color='blue'),
                     fontsize=10, color='blue')
    if hc_cost_progress:
        plt.annotate(f'Final HC Cost: {hc_cost_progress[-1]:,.0f}',
                     xy=(len(hc_cost_progress)-1, hc_cost_progress[-1]),
                     xytext=(len(hc_cost_progress)*0.7, hc_cost_progress[-1]*0.9),
                     arrowprops=dict(arrowstyle='->', color='green'),
                     fontsize=10, color='green')
    plt.title('Cost Function Progress: Simulated Annealing vs Hill Climbing vs Genetic Algorithm', fontsize=15, fontweight='bold')
    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('Total Cost', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=12, frameon=True, shadow=True, title='Algorithm', title_fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cost_progress_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Cost progress plot saved in '{save_dir}/cost_progress_comparison.png' (combined view)")

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