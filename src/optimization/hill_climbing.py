import numpy as np
import random
from dataclasses import dataclass
from optimization.cost import CostFunction

@dataclass
class HillClimbingTimeSlot:
    """
    Represents a time slot for train scheduling in HillClimbingTrainScheduler.
    """
    day_type: str
    time_period: str
    max_slots: int
    current_trains: int
    demand: float
    min_frequency: int = 15  # Minimum frequency in minutes
    max_frequency: int = 30  # Maximum frequency in minutes

class HillClimbingTrainScheduler:
    def __init__(self, passenger_data, time_slots, train_capacity=1000, min_frequency=15, max_frequency=30, min_trains=3, max_iterations=1000, early_stopping_rounds=200):
        """
        passenger_data: Processed passenger flow data
        time_slots: Available time slots data
        train_capacity: Maximum passengers per train
        min_frequency: Minimum frequency between trains in minutes
        max_frequency: Maximum frequency between trains in minutes
        min_trains: Minimum number of trains per hour
        max_iterations: Number of hill climbing iterations
        early_stopping_rounds: Stop if no improvement after this many iterations
        """
        self.passenger_data = passenger_data
        self.time_slots = time_slots
        self.train_capacity = train_capacity
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.min_trains = min_trains
        self.max_iterations = max_iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.slots = {}  # Dictionary to store HillClimbingTimeSlot objects
        self.penalties = {
            'constraint_violation': 1e6,
            'overload': 10,
            'underutil': 100,
            'wait_time': 0.1,
            'frequency': 100,
            'no_trains': 1e8
        }

    def initialize_slots(self, onboard_demand_df=None):
        """
        Initialize time slots with current train allocations (hourly, by direction).
        If onboard_demand_df is provided, use its max_onboard as demand.
        """
        onboard_lookup = None
        if onboard_demand_df is not None:
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
                continue
            max_slots = slot_row['max_slots'].iloc[0]
            default_trains = slot_row['default_trains'].iloc[0]
            demand = onboard_lookup.get(key, row['total_ons'] + row['total_offs']) if onboard_lookup else row['total_ons'] + row['total_offs']
            self.slots[key] = HillClimbingTimeSlot(
                day_type=row['day_type_name'],
                time_period=row['hour'],
                max_slots=max_slots,
                current_trains=default_trains,
                demand=demand,
                min_frequency=self.min_frequency,
                max_frequency=self.max_frequency
            )

    def optimize(self, geo_mean_df):
        """
        Optimize train allocations using hill climbing.
        Args:
            geo_mean_df: DataFrame with geometric mean ons for each (day_type_name, hour, direction_id)
        Returns:
            Best solution and cost progress
        """
        self.initialize_slots()
        
        self.geo_lookup = {(row['day_type_name'], row['hour'], row['direction_id']): row['geo_mean_ons'] for _, row in geo_mean_df.iterrows()}
        
        # Random initialization of train allocations
        for key, slot in self.slots.items():
            min_possible = max(self.min_trains, int(60 / slot.max_frequency))
            max_possible = min(slot.max_slots, int(60 / slot.min_frequency))
            slot.current_trains = random.randint(min_possible, max_possible)
            
        current_cost = CostFunction(self).cost_fn()
        best_solution = {key: slot.current_trains for key, slot in self.slots.items()}
        best_cost = current_cost
        print(f"HC Initial Cost: {current_cost}")
        restart_count = 0
        max_restarts = 3
        no_improve_rounds = 0
        cost_progress = []  # Track cost at each iteration
        
        for restart in range(max_restarts):
            print(f"HC Restart {restart + 1}/{max_restarts}")
            for i in range(self.max_iterations // max_restarts):
                slot_key = random.choice(list(self.slots.keys()))
                slot = self.slots[slot_key]
                old_trains = slot.current_trains
                
                # Modify train allocation
                change = random.choice([-2, -1, 1, 2])  # Allow bigger jumps
                new_trains = max(self.min_trains, min(slot.current_trains + change, slot.max_slots))
                slot.current_trains = new_trains
                
                new_cost = CostFunction(self).cost_fn()
                if new_cost < current_cost:
                    current_cost = new_cost
                    no_improve_rounds = 0
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_solution = {key: slot.current_trains for key, slot in self.slots.items()}
                else:
                    slot.current_trains = old_trains
                    no_improve_rounds += 1
                
                # Track cost progress
                cost_progress.append(current_cost)
                
                if i % 100 == 0:
                    print(f"HC Iteration {i:4d}: Current Cost = {current_cost}, Best Cost = {best_cost:,.2f}")
                if no_improve_rounds >= self.early_stopping_rounds:
                    print(f"Early stopping at iteration {i} due to no improvement.")
                    break
                    
            # Reinitialize for next restart
            if restart < max_restarts - 1:
                for key, slot in self.slots.items():
                    min_possible = max(self.min_trains, int(60 / slot.max_frequency))
                    max_possible = min(slot.max_slots, int(60 / slot.min_frequency))
                    slot.current_trains = random.randint(min_possible, max_possible)
                current_cost = CostFunction(self).cost_fn()
                
        print(f"HC Final Best Cost: {best_cost:,.2f}")
        
        # Set slots to best solution
        for key, trains in best_solution.items():
            self.slots[key].current_trains = trains
            
        return best_solution, cost_progress
