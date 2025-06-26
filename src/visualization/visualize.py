import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json
import os
from datetime import datetime, timedelta

class ScheduleVisualizer:
    def __init__(self, passenger_data: pd.DataFrame, optimization_results: Dict[str, int]):
        """
        Initialize the schedule visualizer.
        
        Args:
            passenger_data: Processed passenger flow data
            optimization_results: Dictionary of optimized train allocations
        """
        self.passenger_data = passenger_data
        self.optimization_results = optimization_results
        
    def plot_demand_distribution(self, save_path: str = None, algorithm_name: str = ""):
        """
        Plot normalized (per-day) passenger demand distribution across hours and directions.
        Args:
            save_path: Optional path to save the plot
            algorithm_name: Optional name of the algorithm used for optimization
        """
        plt.figure(figsize=(16, 8))

        # Define service day counts
        service_days = {
            'weekday': 77,
            'saturday': 12,
            'sunday': 8
        }

        # Aggregate demand
        demand_data = self.passenger_data.groupby(['day_type_name', 'hour', 'direction_id']).agg({
            'total_ons': 'sum',
            'total_offs': 'sum'
        }).reset_index()

        # Normalize total_ons using number of service days
        demand_data['normalized_ons'] = demand_data.apply(
            lambda row: row['total_ons'] / service_days.get(row['day_type_name'], 1),
            axis=1
        )

        # Plot normalized demand
        sns.barplot(
            data=demand_data,
            x='hour',
            y='normalized_ons',
            hue='day_type_name',
            errorbar=None
        )

        title = 'Passenger Demand Distribution per Day by Hour and Direction'
        if algorithm_name:
            title += f" ({algorithm_name})"
        plt.title(title)
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Daily Passengers')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()

        
    def plot_train_allocation(self, save_path: str = None, algorithm_name: str = ""):
        """
        Plot optimized train allocation across hours and directions.
        Args:
            save_path: Optional path to save the plot
            algorithm_name: Optional name of the algorithm used for optimization
        """
        plt.figure(figsize=(16, 8))
        # Convert optimization results to DataFrame
        allocation_data = []
        for key, trains in self.optimization_results.items():
            day_type, hour, direction = key
            allocation_data.append({
                'day_type_name': day_type,
                'hour': hour,
                'direction_id': direction,
                'trains': trains
            })
        allocation_df = pd.DataFrame(allocation_data)
        # Create grouped bar plot
        sns.barplot(
            data=allocation_df,
            x='hour',
            y='trains',
            hue='day_type_name',
            errorbar=None
        )
        title = 'Optimized Train Allocation by Hour and Direction'
        if algorithm_name:
            title += f" ({algorithm_name})"
        plt.title(title)
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Trains')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_load_distribution(self, train_capacity: int, save_path: str = None, algorithm_name: str = ""):
        """
        Plot passenger load distribution per train by hour and direction.
        Args:
            train_capacity: Maximum capacity per train
            save_path: Optional path to save the plot
            algorithm_name: Optional name of the algorithm used for optimization
        """
        plt.figure(figsize=(16, 8))
        # Calculate load per train
        load_data = []
        for key, trains in self.optimization_results.items():
            day_type, hour, direction = key
            slot_df = self.passenger_data[
                (self.passenger_data['day_type_name'] == day_type) &
                (self.passenger_data['hour'] == hour) &
                (self.passenger_data['direction_id'] == direction)
            ]
            # Use adjusted_max_onboard if available, else max_onboard, else total_ons
            if 'adjusted_max_onboard' in slot_df.columns and not slot_df.empty:
                demand = slot_df['adjusted_max_onboard'].iloc[0]
            elif 'max_onboard' in slot_df.columns and not slot_df.empty:
                demand = slot_df['max_onboard'].iloc[0]
            else:
                demand = slot_df['total_ons'].sum()
            if trains > 0:
                load_per_train = demand / trains
                load_data.append({
                    'day_type_name': day_type,
                    'hour': hour,
                    'direction_id': direction,
                    'load_per_train': load_per_train
                })
        load_df = pd.DataFrame(load_data)
        if load_df.empty:
            print("No load distribution data to plot (all train allocations are zero or missing). Skipping plot.")
            return
        # Create grouped bar plot
        sns.barplot(
            data=load_df,
            x='hour',
            y='load_per_train',
            hue='day_type_name',
            errorbar=None
        )
        # Add capacity line
        plt.axhline(y=train_capacity, color='r', linestyle='--', label='Train Capacity')
        title = 'Passenger Load per Train by Hour and Direction'
        if algorithm_name:
            title += f" ({algorithm_name})"
        plt.title(title)
        plt.xlabel('Hour of Day')
        plt.ylabel('Passengers per Train')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def generate_report(self, train_capacity: int, output_path: str):
        """
        Generate a comprehensive report of the optimization results (hourly, by direction).
        Args:
            train_capacity: Maximum capacity per train
            output_path: Path to save the report
        """
        report = {
            'optimization_results': self.optimization_results,
            'summary_statistics': {}
        }
        # Calculate summary statistics
        for key, trains in self.optimization_results.items():
            day_type, hour, direction = key
            demand = self.passenger_data[
                (self.passenger_data['day_type_name'] == day_type) &
                (self.passenger_data['hour'] == hour) &
                (self.passenger_data['direction_id'] == direction)
            ]['total_ons'].sum()
            load_per_train = demand / trains if trains > 0 else 0
            utilization = (load_per_train / train_capacity) * 100 if trains > 0 else 0
            report['summary_statistics'][key] = {
                'trains_allocated': int(trains),
                'total_demand': int(demand),
                'load_per_train': int(load_per_train),
                'capacity_utilization': float(round(utilization, 2))
            }
        # Convert tuple keys to strings for JSON serialization
        report['optimization_results'] = {
            '|'.join(map(str, key)): int(value) for key, value in self.optimization_results.items()
        }
        report['summary_statistics'] = {
            '|'.join(map(str, key)): {k: (int(v) if isinstance(v, (np.integer, int)) else float(v)) for k, v in value.items()} for key, value in report['summary_statistics'].items()
        }
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)

    def plot_train_allocation_comparison(self, other_results: dict, third_results: dict = None, 
                                       other_label: str = "Hill Climbing", this_label: str = "Simulated Annealing", 
                                       third_label: str = "Genetic Algorithm", save_path: str = None):
        """
        Plot side-by-side comparison of train allocations for two or three optimization results,
        focusing on total trains per hour for weekdays.
        Args:
            other_results: Dictionary of train allocations from the second optimizer
            third_results: Dictionary of train allocations from the third optimizer (optional)
            other_label: Label for the second optimizer
            this_label: Label for this optimizer
            third_label: Label for the third optimizer
            save_path: Optional path to save the plot
        """
        # Convert results to DataFrames
        def to_df(results, label):
            data = []
            for key, trains in results.items():
                day_type, hour, direction = key
                data.append({
                    'day_type_name': day_type,
                    'hour': hour,
                    'direction_id': direction,
                    'trains': trains,
                    'optimizer': label
                })
            return pd.DataFrame(data)

        df1 = to_df(self.optimization_results, this_label)
        df2 = to_df(other_results, other_label)
        
        if third_results is not None:
            df3 = to_df(third_results, third_label)
            combined = pd.concat([df1, df2, df3], ignore_index=True)
        else:
            combined = pd.concat([df1, df2], ignore_index=True)

        # Filter for weekday and sum trains per hour
        weekday_comparison = combined[combined['day_type_name'] == 'weekday'].copy()
        hourly_totals = weekday_comparison.groupby(['hour', 'optimizer'])['trains'].sum().reset_index()

        # Plot
        plt.figure(figsize=(18, 8))
        sns.barplot(
            data=hourly_totals,
            x='hour',
            y='trains',
            hue='optimizer'
        )
        title = 'Train Allocation Comparison by Hour (Weekday Total)'
        if third_results is not None:
            title += f" - {this_label} vs {other_label} vs {third_label}"
        else:
            title += f" - {this_label} vs {other_label}"
        plt.title(title)
        plt.xlabel('Hour of Day')
        plt.ylabel('Total Number of Trains per Hour')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def generate_train_schedule(self, save_path: str = None):
        """
        Generate a schedule table showing train departure times from Forest Hills (direction 0)
        and Oak Grove (direction 1) based on optimized frequencies.
        Args:
            save_path: Optional path to save the schedule as CSV
        """
        schedule_data = []
        
        # Process each hour and direction
        for key, trains in self.optimization_results.items():
            day_type, hour, direction = key
            if trains > 0:  # Only process hours with trains
                # Calculate minutes between trains
                minutes_between_trains = 60 / trains
                
                # Generate departure times for this hour
                for train_num in range(trains):
                    # Calculate departure minute
                    departure_minute = int(train_num * minutes_between_trains)
                    
                    # Format time as HH:MM
                    departure_time = f"{hour:02d}:{departure_minute:02d}"
                    
                    # Determine station name based on direction
                    station = "Forest Hills" if direction == 0 else "Oak Grove"
                    
                    schedule_data.append({
                        'Day Type': day_type,
                        'Hour': hour,
                        'Direction': direction,
                        'Station': station,
                        'Departure Time': departure_time,
                        'Trains per Hour': trains,
                        'Minutes Between Trains': round(minutes_between_trains, 1)
                    })
        
        # Convert to DataFrame and sort
        schedule_df = pd.DataFrame(schedule_data)
        schedule_df = schedule_df.sort_values(['Day Type', 'Hour', 'Direction', 'Departure Time'])
        
        # Save to CSV if path provided
        if save_path:
            schedule_df.to_csv(save_path, index=False)
            
        return schedule_df

    def generate_simple_schedule(self, day_type: str = 'weekday', save_path: str = None):
        """
        Generate a simple schedule table showing only station and departure time, starting from 6:00 AM.
        Args:
            day_type: Optional filter for a specific day type (e.g., 'weekday'). If None, include all.
            save_path: Optional path to save the schedule as CSV
        Returns:
            schedule_df: DataFrame with columns ['Station', 'Departure Time']
        """
        schedule_data = []
        for key, trains in self.optimization_results.items():
            d_type, hour, direction = key
            if hour < 6:
                continue
            if day_type and d_type != day_type:
                continue
            if trains > 0:
                minutes_between_trains = 60 / trains
                for train_num in range(trains):
                    departure_minute = int(train_num * minutes_between_trains)
                    departure_time = f"{hour:02d}:{departure_minute:02d}"
                    station = "Forest Hills" if direction == 0 else "Oak Grove"
                    schedule_data.append({
                        'Station': station,
                        'Departure Time': departure_time
                    })
        schedule_df = pd.DataFrame(schedule_data)
        # Deduplicate by Station and Departure Time
        schedule_df = schedule_df.drop_duplicates(subset=['Station', 'Departure Time'])
        schedule_df = schedule_df.sort_values(['Station', 'Departure Time'])
        if save_path:
            schedule_df.to_csv(save_path, index=False)
        return schedule_df

def generate_full_eta_table(schedule_csv_path, distances_csv_path, output_csv_path):
    ORANGE_LINE_STATIONS = [
        "Oak Grove", "Malden Center", "Wellington", "Assembly", "Sullivan Square", "Community College",
        "North Station", "Haymarket", "State", "Downtown Crossing", "Chinatown", "Tufts Medical Center",
        "Back Bay", "Massachusetts Avenue", "Ruggles", "Roxbury Crossing", "Jackson Square", "Stony Brook",
        "Green Street", "Forest Hills"
    ]

    schedule_df = pd.read_csv(schedule_csv_path)
    distances_df = pd.read_csv(distances_csv_path)
    distances_df = distances_df[distances_df['route_id'] == 'Orange']

    def build_travel_time_lookup():
        lookup = {0: {}, 1: {}}
        for direction in [0, 1]:
            for _, row in distances_df[distances_df['direction_id'] == direction].iterrows():
                from_stop = row['from_stop_name']
                to_stop = row['to_stop_name']
                time = row['from_to_meters'] / 7.73379 / 60 + 1
                lookup[direction][(from_stop, to_stop)] = time
        return lookup
    travel_time_lookup = build_travel_time_lookup()

    eta_rows = []
    for idx, row in schedule_df.iterrows():
        origin = row['Station']
        dep_time_str = row['Departure Time']
        dep_time = datetime.strptime(dep_time_str, "%H:%M")
        if origin == "Forest Hills":
            direction = 1
            stops = ORANGE_LINE_STATIONS
        else:
            direction = 0
            stops = ORANGE_LINE_STATIONS
        curr_time = dep_time
        eta_rows.append({"Train Origin": origin, "Departure Time": dep_time_str, "Stop": stops[0], "ETA": curr_time.strftime("%H:%M")})
        for i in range(len(stops)-1):
            seg = (stops[i], stops[i+1])
            travel_min = travel_time_lookup[direction].get(seg, 0)
            curr_time = curr_time + timedelta(minutes=travel_min)
            eta_rows.append({"Train Origin": origin, "Departure Time": dep_time_str, "Stop": stops[i+1], "ETA": curr_time.strftime("%H:%M")})
    eta_df = pd.DataFrame(eta_rows)
    eta_df.to_csv(output_csv_path, index=False)
    return eta_df

def build_terminal_to_station_lookup(distances_csv_path, output_csv_path):
    import pandas as pd
    df = pd.read_csv(distances_csv_path)
    df = df[df['route_id'] == 'Orange']
    speed = 7.73379  # meters per second
    df['travel_time_minutes'] = df['from_to_meters'] / speed / 60
    df['total_time_minutes'] = df['travel_time_minutes'] + 1
    df = df.sort_values(['direction_id', 'cumulative_meters'])
    df['cumulative_time'] = df.groupby(['direction_id'])['total_time_minutes'].cumsum()
    # Get all unique stops in order for each direction
    stops_0 = [df[df['direction_id'] == 0].iloc[0]['from_stop_name']] + list(df[df['direction_id'] == 0]['to_stop_name'])
    stops_1 = [df[df['direction_id'] == 1].iloc[0]['from_stop_name']] + list(df[df['direction_id'] == 1]['to_stop_name'])
    # For direction 0
    cum_times_0 = [0]
    for stop in stops_0[1:]:
        cum_times_0.append(df[(df['direction_id'] == 0) & (df['to_stop_name'] == stop)]['cumulative_time'].values[0])
    # For direction 1
    cum_times_1 = [0]
    for stop in stops_1[1:]:
        cum_times_1.append(df[(df['direction_id'] == 1) & (df['to_stop_name'] == stop)]['cumulative_time'].values[0])
    # Build DataFrame
    lookup_df = pd.DataFrame({
        'Station': stops_1,  # Both lists have the same stations, just reversed
        'From Forest Hills (min)': cum_times_1,
        'From Oak Grove (min)': cum_times_0[::-1]  # Reverse to match station order
    })
    lookup_df.to_csv(output_csv_path, index=False)
    return lookup_df

# Only run if called directly (not on import)
if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append("..")
    from data_processing.preprocess import DataPreprocessor
    from optimization.optimize import TrainScheduler
    
    # Load and process data
    preprocessor = DataPreprocessor(
        passenger_data_path="data/passenger_flow/passenger_data.csv",
        gtfs_data_path="data/gtfs"
    )
    passenger_data, time_slots = preprocessor.process_data()
    
    # Create scheduler and optimize
    scheduler = TrainScheduler(passenger_data, time_slots)
    optimization_results = scheduler.simulated_annealing()
    
    # Create visualizer and generate plots
    visualizer = ScheduleVisualizer(passenger_data, optimization_results)
    visualizer.plot_demand_distribution('plots/demand_distribution.png')
    visualizer.plot_train_allocation('plots/train_allocation.png')
    visualizer.plot_load_distribution(1000, 'plots/load_distribution.png')
    visualizer.generate_report(1000, 'reports/optimization_report.json')

    # Generate the full ETA table for use in the Streamlit app
    schedule_csv = os.path.join(os.path.dirname(__file__), '../../reports/simple_schedule_sa.csv')
    distances_csv = os.path.join(os.path.dirname(__file__), '../../MBTA_Rapid_Transit_Stop_Distances.csv')
    output_csv = os.path.join(os.path.dirname(__file__), '../../reports/full_eta_table.csv')
    generate_full_eta_table(schedule_csv, distances_csv, output_csv) 