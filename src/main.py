import sys
from pathlib import Path
import pandas as pd
from data_processing.preprocess import DataPreprocessor
from optimization.simulated_annealing import SimulatedAnnealingTrainScheduler
from optimization.hill_climbing import HillClimbingTrainScheduler
from optimization.genetic_algorithm import GeneticAlgorithmScheduler
from visualization.visualize import ScheduleVisualizer
from plot_costs import plot_cost_progress, plot_combined_cost_progress

def main():
    # Create necessary directories
    Path("plots").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    
    print("Loading and preprocessing data...")
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(
        passenger_data_path="data/passenger_flow/passenger_Data.csv",
        gtfs_data_path="data/gtfs"
    )
    
    # Process data (now returns 3 values)
    passenger_data, time_slots, geo_mean_demand = preprocessor.process_data()

    # Compute realistic onboard demand for each group
    onboard_demand_df = preprocessor.compute_onboard_demand(passenger_data)
    print('onboard_demand_df columns:', onboard_demand_df.columns.tolist())  # Debug print
    onboard_hourly_df = onboard_demand_df  # Already has 'hour', 'day_type_name', 'direction_id', 'max_onboard'

    print("\nInitializing train scheduler...")
    # Initialize train scheduler with custom parameters
    scheduler = SimulatedAnnealingTrainScheduler(
        passenger_data=passenger_data,
        time_slots=time_slots,
        train_capacity=1000,    # Maximum passengers per train
        min_frequency=6,        # Minimum 6 minutes between trains
        max_frequency=15        # Maximum 15 minutes between trains
    )
    # Use improved demand modeling
    scheduler.initialize_slots(onboard_demand_df=onboard_hourly_df)

    print("\nOptimizing train frequency with Simulated Annealing...")
    sa_solution, sa_cost_progress = scheduler.optimize_frequency(geo_mean_demand, max_iterations=1000)
    print("\nSimulated Annealing Frequency Analysis:")
    sa_analysis = scheduler.get_frequency_analysis(sa_solution)
    print(sa_analysis.to_string(index=False))
    sa_analysis.to_csv("reports/sa_frequency_analysis.csv", index=False)
    
    print("\nOptimizing train frequency with Hill Climbing...")
    
    #initialize Hill Climbing scheduler 

    scheduler.initialize_slots(onboard_demand_df=onboard_hourly_df)  
    hc_optimizer = HillClimbingTrainScheduler(
        passenger_data=passenger_data,
        time_slots=time_slots,
        train_capacity=1000,
        min_frequency=6,
        max_frequency=15,
        min_trains=3,
        max_iterations=1000
    )
    hc_solution, hc_cost_progress = hc_optimizer.optimize(geo_mean_demand)  
    print("\nHill Climbing Frequency Analysis:")
    hc_analysis = scheduler.get_frequency_analysis(hc_solution)
    print(hc_analysis.to_string(index=False))
    hc_analysis.to_csv("reports/hc_frequency_analysis.csv", index=False)
    
    print("\nOptimizing train frequency with Genetic Algorithm...")
    ga_optimizer = GeneticAlgorithmScheduler(
        slots=scheduler.slots,  # Pass slots directly
        geo_mean_df=geo_mean_demand,
        penalties=scheduler.penalties,  # Pass penalties directly
        train_capacity=1000,
        min_trains=3,
        population_size=50,
        generations=1000,
        mutation_rate=0.1,
        elite_size=5,
        crossover_rate=0.8
    )
    ga_solution, ga_cost, ga_cost_progress = ga_optimizer.optimize()
    print("\nGenetic Algorithm Frequency Analysis:")
    ga_analysis = scheduler.get_frequency_analysis(ga_solution)
    print(ga_analysis.to_string(index=False))
    ga_analysis.to_csv("reports/ga_frequency_analysis.csv", index=False)

    print("\nGenerating visualizations for Simulated Annealing solution...")
    
    #Intialize visualizer with Simulated Annealing solution
    sa_visualizer = ScheduleVisualizer(passenger_data, sa_solution,"Simulated Annealing")
    sa_visualizer.plot_demand_distribution("plots/demand_distribution.png")
    sa_visualizer.plot_train_allocation("plots/train_allocation_sa.png")
    sa_visualizer.plot_load_distribution(1000, "plots/load_distribution_sa.png")
    sa_visualizer.generate_report(1000, "reports/optimization_report.json")
    
    #Intialize visualizer with Simulated Annealing solution

    print("\nGenerating visualizations for Hill Climbing solution...")
    hc_visualizer = ScheduleVisualizer(passenger_data, hc_solution,"Hill Climbing")
    hc_visualizer.plot_train_allocation("plots/train_allocation_hc.png")
    hc_visualizer.plot_load_distribution(1000, "plots/load_distribution_hc.png")

    print("\nGenerating visualizations for Genetic Algorithm solution...")
    ga_visualizer = ScheduleVisualizer(passenger_data, ga_solution, "Genetic Algorithm")
    ga_visualizer.plot_train_allocation("plots/train_allocation_ga.png")
    ga_visualizer.plot_load_distribution(1000, "plots/load_distribution_ga.png")

    # Plot comparison of train allocations
    print("\nPlotting side-by-side comparison of train allocations...")
    sa_visualizer.plot_train_allocation_comparison(hc_solution, other_label="Hill Climbing", this_label="Simulated Annealing", save_path="plots/train_allocation_comparison.png")
    sa_visualizer.plot_train_allocation_comparison(
        ga_solution, other_label="Genetic Algorithm", this_label="Simulated Annealing", save_path="plots/train_allocation_comparison_ga.png"
    )

    print("\nGenerating train schedule...")
    schedule_df = sa_visualizer.generate_train_schedule(save_path="reports/train_schedule.csv")
    
    # Display schedule for each day type
    for day_type in schedule_df['Day Type'].unique():
        print(f"\n{day_type.upper()} SCHEDULE:")
        day_schedule = schedule_df[schedule_df['Day Type'] == day_type]
        print(day_schedule[['Station', 'Departure Time', 'Trains per Hour', 'Minutes Between Trains']].to_string(index=False))
    
    # Generate and display simple schedule for Simulated Annealing (from 6:00 AM, weekday only, deduplicated)
    print("\nSimple Train Schedule (Simulated Annealing, from 6:00 AM, weekday only):")
    simple_schedule = sa_visualizer.generate_simple_schedule(day_type='weekday', save_path="reports/simple_schedule_sa.csv")
    print(simple_schedule.head(20))
    
    print("\nOptimization complete! Results saved in 'plots' and 'reports' directories.")

    # Plot cost progress
    print("\nGenerating cost progress plots...")
    plot_cost_progress(sa_cost_progress, hc_cost_progress, ga_cost_progress, "plots")
    plot_combined_cost_progress(sa_cost_progress, hc_cost_progress, ga_cost_progress, "plots")

if __name__ == "__main__":
    main()