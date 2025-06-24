import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def plot_cost_progress(sa_cost_progress, hc_cost_progress, ga_cost_progress, save_dir="plots"):
    """
    Plot cost function progress for Simulated Annealing, Hill Climbing and Genetic Algorithm.
    
    Args:
        sa_cost_progress: List of costs from Simulated Annealing optimization
        hc_cost_progress: List of costs from Hill Climbing optimization
        ga_cost_progress: List of costs from Genetic Algorithm optimization 
        save_dir: Directory to save the plots
    """
    # Create plots directory if it doesn't exist
    Path(save_dir).mkdir(exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3 if ga_cost_progress else 2, 1, figsize=(12, 15 if ga_cost_progress else 10))
    
    # Plot 1: Simulated Annealing Cost Progress
    iterations_sa = range(len(sa_cost_progress))
    ax1 = axes[0]
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
    ax2 = axes[1]
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
    
    # Plot 3: Genetic Algorithm Cost Progress (if provided)
    if ga_cost_progress:
        iterations_ga = range(len(ga_cost_progress))
        ax3 = axes[2]
        ax3.plot(iterations_ga, ga_cost_progress, 'r-', linewidth=1.5, label='Current Cost')
        ax3.set_title('Genetic Algorithm - Cost Function Progress', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Total Cost', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Add final cost annotation for GA
        final_cost_ga = ga_cost_progress[-1] if ga_cost_progress else 0
        ax3.annotate(f'Final Cost: {final_cost_ga:,.0f}', 
                    xy=(len(ga_cost_progress)-1, final_cost_ga),
                    xytext=(len(ga_cost_progress)*0.7, final_cost_ga*1.2),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cost_progress_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Cost progress plots saved in '{save_dir}' directory:")
    print(f"- {save_dir}/cost_progress_comparison.png (combined view)")
    
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
    
    # Individual GA plot (if provided)
    if ga_cost_progress:
        plt.figure(figsize=(10, 6))
        plt.plot(iterations_ga, ga_cost_progress, 'r-', linewidth=2)
        plt.title('Genetic Algorithm - Cost Function Progress', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Total Cost', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_dir}/ga_cost_progress.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"- {save_dir}/ga_cost_progress.png (Genetic Algorithm only)")

def plot_combined_cost_progress(sa_cost_progress, hc_cost_progress, ga_cost_progress, save_dir="plots"):
    """
    Plot combined cost function progress for Simulated Annealing, Hill Climbing, and Genetic Algorithm.
    
    Args:
        sa_cost_progress: List of costs from Simulated Annealing optimization
        hc_cost_progress: List of costs from Hill Climbing optimization
        ga_cost_progress: List of costs from Genetic Algorithm optimization
        save_dir: Directory to save the plot
    """
    # Create plots directory if it doesn't exist
    Path(save_dir).mkdir(exist_ok=True)
    
    # Determine the maximum number of iterations across all methods
    max_iterations = max(len(sa_cost_progress), len(hc_cost_progress), len(ga_cost_progress))
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot Simulated Annealing
    iterations_sa = range(len(sa_cost_progress))
    plt.plot(iterations_sa, sa_cost_progress, 'b-', linewidth=1.5, label='Simulated Annealing')
    
    # Plot Hill Climbing
    iterations_hc = range(len(hc_cost_progress))
    plt.plot(iterations_hc, hc_cost_progress, 'g-', linewidth=1.5, label='Hill Climbing')
    
    # Plot Genetic Algorithm
    iterations_ga = range(len(ga_cost_progress))
    plt.plot(iterations_ga, ga_cost_progress, 'r-', linewidth=1.5, label='Genetic Algorithm')
    
    # Add labels, title, and legend
    plt.title('Combined Cost Function Progress', fontsize=16, fontweight='bold')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Total Cost', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{save_dir}/combined_cost_progress.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Combined cost progress plot saved in '{save_dir}/combined_cost_progress.png'")

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
    ga_sample = [10000, 9700, 9400, 9100, 8800, 8500, 8200, 8100, 8000, 7800]
    
    print("\nPlotting sample cost progress data...")
    plot_cost_progress(sa_sample, hc_sample, ga_sample)
    plot_combined_cost_progress(sa_sample, hc_sample, ga_sample)

if __name__ == "__main__":
    main()