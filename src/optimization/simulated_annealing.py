"""
Simulated Annealing Algorithm for Train Scheduling Optimization

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import random

@dataclass
class TimeSlot:
    day_type: str
    time_period: str
    max_slots: int
    current_trains: int
    demand: float
    min_frequency: int = 6  # Minimum frequency in minutes
    max_frequency: int = 30  # Maximum frequency in minutes

class TrainScheduler:
    """
    Simulated Annealing Scheduler for optimizing train allocation.
    
   
    """
    
    def __init__(self, 
                 passenger_data: pd.DataFrame,
                 time_slots: pd.DataFrame,
                 train_capacity: int = 1000,
                 min_frequency: int = 6,
                 max_frequency: int = 30):
        """
        Initialize the train scheduler.
        
        parameters:
            passenger_data: Processed passenger flow data
            time_slots: Available time slots data
            train_capacity: Maximum capacity per train
            min_frequency: Minimum frequency between trains in minutes
            max_frequency: Maximum frequency between trains in minutes
        """
        self.passenger_data = passenger_data
        self.time_slots = time_slots
        self.train_capacity = train_capacity
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.slots: Dict[tuple, TimeSlot] = {}
        # Penalty configuration
        self.penalties = {
            'constraint_violation': 1e6,
            'overload': 10,
            'underutil': 100,
            'wait_time': 0.1,
            'frequency': 100,
            'no_trains': 1e8
        }
        self.geo_lookup = None  # Will be set in optimize_frequency
        
    def initialize_slots(self, onboard_demand_df=None):
        """
        Initialize time slots with current train allocations (hourly, by direction).
        If onboard_demand_df is provided, use its max_onboard as demand.
        """
        onboard_lookup = None
        if onboard_demand_df is not None:
            # Create a lookup dict: (day_type_name, hour, direction_id) -> max_onboard
            onboard_lookup = {
                (row['day_type_name'], row['hour'], row['direction_id']): row['max_onboard']
                for _, row in onboard_demand_df.iterrows()
            }
        for _, row in self.passenger_data.groupby(['day_type_name', 'hour', 'direction_id']).agg({
            'total_ons': 'sum',
            'total_offs': 'sum'
        }).reset_index().iterrows():
            key = (row['day_type_name'], row['hour'], row['direction_id'])
            slot_row = self.time_slots[
                (self.time_slots['day_type_name'] == row['day_type_name']) &
                (self.time_slots['hour'] == row['hour']) &
                (self.time_slots['direction_id'] == row['direction_id'])
            ]
            if slot_row.empty:
                continue  # skip this slot if not found
            max_slots = slot_row['max_slots'].iloc[0]
            default_trains = slot_row['default_trains'].iloc[0]
            # Use max_onboard as demand if available, else fallback to total_ons + total_offs
            demand = None
            if onboard_lookup is not None:
                demand = onboard_lookup.get(key, 0)
            if demand is None:
                demand = row['total_ons'] + row['total_offs']
            self.slots[key] = TimeSlot(
                day_type=row['day_type_name'],
                time_period=row['hour'],  # now hour
                max_slots=max_slots,
                current_trains=default_trains,
                demand=demand,
                min_frequency=self.min_frequency,
                max_frequency=self.max_frequency
            )
    
    def calculate_cost(self) -> float:
        """
        Calculate the current cost of the schedule.
        Cost is based on:
        1. Wait time for passengers
        2. Train capacity utilization
        3. Frequency constraints
        
        Returns:
            Total cost of the current schedule
        """
        total_cost = 0
        for slot in self.slots.values():
            if slot.current_trains > 0:
                #  average wait time 
                frequency = 60 / slot.current_trains  # minutes between trains
                wait_time = frequency / 2
                
                #  load per train
                load_per_train = slot.demand / slot.current_trains
                
                # Cost components
                wait_time_cost = wait_time * slot.demand * self.penalties['wait_time']
                capacity_cost = (load_per_train - self.train_capacity) ** 2
                
                # Frequency constraint cost
                if frequency < slot.min_frequency:
                    frequency_cost = (slot.min_frequency - frequency) * self.penalties['frequency']
                elif frequency > slot.max_frequency:
                    frequency_cost = (frequency - slot.max_frequency) * self.penalties['frequency']
                else:
                    frequency_cost = 0
                
                total_cost += wait_time_cost + capacity_cost + frequency_cost
                
        return total_cost
    
    def optimize_frequency(self, geo_mean_df, train_capacity=1000, min_trains=3, max_iterations: int = 1000) -> dict:
        """
        Use simulated annealing local search to optimize train allocation per hour.
        
        parameters:
            geo_mean_df: DataFrame with geometric mean ons for each (day_type_name, hour, direction_id)
            train_capacity: Maximum passengers per train
            min_trains: Minimum number of trains per hour
            max_iterations: Number of local search iterations
            
        Returns:
            Dictionary of optimized train allocations
        """
        self.initialize_slots()
        
        # Preprocess geo_mean_df into a dict 
        self.geo_lookup = {(row['day_type_name'], row['hour'], row['direction_id']): row['geo_mean_ons'] for _, row in geo_mean_df.iterrows()}
        
        # Start with random allocation 
        for key, slot in self.slots.items():
            # Random initial allocation within constraints
            min_possible = max(min_trains, int(60 / slot.max_frequency))
            max_possible = min(slot.max_slots, int(60 / slot.min_frequency))
            slot.current_trains = random.randint(min_possible, max_possible)
        
        # Improved cost function with better gradients
        def cost_fn():
            total_cost = 0
            for key, slot in self.slots.items():
                day_type, hour, direction = key
                geo_mean_ons = self.geo_lookup.get((day_type, hour, direction), 0)
                trains = slot.current_trains
                
                # Constraint violations
                if trains < min_trains or trains > slot.max_slots:
                    total_cost += self.penalties['constraint_violation']
                    continue
                
                if trains > 0:
                    # Load-based cost with smoother gradients
                    load_per_train = geo_mean_ons / trains
                    
                    # Overload penalty 
                    overload = max(0, load_per_train - train_capacity)
                    total_cost += overload**2 * self.penalties['overload']
                    
                    # Underutilization penalty 
                    underutil = max(0, train_capacity * 0.3 - load_per_train)  # Penalize if less than 30% full
                    total_cost += underutil * self.penalties['underutil']
                    
                    # Wait time cost 
                    frequency = 60 / trains
                    wait_time = frequency / 2
                    total_cost += wait_time * geo_mean_ons * self.penalties['wait_time']
                    
                    # Frequency constraint cost 
                    if frequency < slot.min_frequency:
                        total_cost += (slot.min_frequency - frequency) * self.penalties['frequency']
                    elif frequency > slot.max_frequency:
                        total_cost += (frequency - slot.max_frequency) * self.penalties['frequency']
                else:
                    # Heavy penalty for no trains when there's demand
                    if geo_mean_ons > 0:
                        total_cost += self.penalties['no_trains']
                        
            return total_cost
        
        current_cost = cost_fn()
        best_solution = {key: slot.current_trains for key, slot in self.slots.items()}
        best_cost = current_cost
        
        # Simulated annealing parameters
        temperature = 1000.0  # Higher initial temperature for more exploration
        cooling_rate = 0.995  # Slower cooling
        
        print(f"SA Initial Cost: {current_cost:,.2f}")
        
        cost_progress = []  # Track cost at each iteration
        
        for i in range(max_iterations):
            # Generate neighbor by modifying multiple slots
            num_changes = random.randint(1, 3)  # Change 1-3 slots at once
            old_trains = {}
            
            for _ in range(num_changes):
                slot_key = random.choice(list(self.slots.keys()))
                slot = self.slots[slot_key]
                old_trains[slot_key] = slot.current_trains
                
                # More aggressive changes
                change = random.choice([-2, -1, 1, 2])  # Allow bigger jumps
                new_trains = slot.current_trains + change
                new_trains = max(min_trains, min(new_trains, slot.max_slots))
                slot.current_trains = new_trains
            
            new_cost = cost_fn()
            
            # Acceptance probability 
            cost_diff = new_cost - current_cost
            if cost_diff < 0 or random.random() < np.exp(-cost_diff / temperature):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = {key: slot.current_trains for key, slot in self.slots.items()}
            else:
                # Revert changes
                for slot_key, old_train_count in old_trains.items():
                    self.slots[slot_key].current_trains = old_train_count
            
            # Track cost progress
            cost_progress.append(current_cost)
            
            temperature *= cooling_rate
            
            if i % 100 == 0:
                print(f"SA Iteration {i:4d}: Current Cost = {current_cost:,.2f}, Best Cost = {best_cost:,.2f}, Temp = {temperature:.2f}")
        
        print(f"SA Final Best Cost: {best_cost:,.2f}")
        
        # Set slots to best solution
        for key, trains in best_solution.items():
            self.slots[key].current_trains = trains
        
        # Strictly enforce frequency after optimization
        for key, slot in self.slots.items():
            frequency = 60 / slot.current_trains if slot.current_trains > 0 else float('inf')
            if frequency < slot.min_frequency:
                slot.current_trains = max(slot.current_trains - 1, min_trains)
            elif frequency > slot.max_frequency:
                slot.current_trains = min(slot.current_trains + 1, slot.max_slots)
        
        return best_solution, cost_progress
    # frequency analysis
    def get_frequency_analysis(self, solution: dict) -> pd.DataFrame:
        analysis_data = []
        for key, trains in solution.items():
            day_type, hour, direction = key
            slot = self.slots[key]
            frequency = 60 / trains if trains > 0 else float('inf')
            load_per_train = slot.demand / trains if trains > 0 else 0
            analysis_data.append({
                'day_type_name': day_type,
                'hour': hour,
                'direction_id': direction,
                'trains': trains,
                'frequency_minutes': round(frequency, 1),
                'load_per_train': round(load_per_train, 0),
                'demand': slot.demand
            })
        return pd.DataFrame(analysis_data)

