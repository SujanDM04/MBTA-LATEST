"""
Hill Climbing Algorithm for Train Scheduling Optimization

This module implements a hill climbing approach to optimize train scheduling
by iteratively improving solutions through local search with restarts.
"""

import numpy as np
import random

class HillClimbingTrainScheduler:
    """
    Hill Climbing Scheduler for optimizing train allocation.
    
    Implements a hill climbing algorithm with multiple restarts to find
    optimal train schedules by making incremental improvements to solutions.
    Uses the same cost function as simulated annealing for fair comparison.
    """
    
    def __init__(self, train_scheduler, geo_mean_df, train_capacity=1000, min_trains=3, max_iterations=1000, early_stopping_rounds=200):
        """
        Initialize the Hill Climbing Scheduler.
        
        parameters:
            train_scheduler: an instance of TrainScheduler (with slots initialized)
            geo_mean_df: DataFrame with geometric mean ons for each (day_type_name, hour, direction_id)
            train_capacity: Maximum passengers per train
            min_trains: Minimum number of trains per hour
            max_iterations: Number of hill climbing iterations
            early_stopping_rounds: Stop if no improvement after this many iterations
        """
        self.train_scheduler = train_scheduler
        self.train_capacity = train_capacity
        self.min_trains = min_trains
        self.max_iterations = max_iterations
        self.early_stopping_rounds = early_stopping_rounds
        # Use same penalty configuration as simulated annealing for fair comparison
        self.penalties = {
            'constraint_violation': 1e6,
            'overload': 10,
            'underutil': 100,
            'wait_time': 0.1,
            'frequency': 100,
            'no_trains': 1e8
        }
        # Preprocess geo_mean_df into a dict 
        self.geo_lookup = {(row['day_type_name'], row['hour'], row['direction_id']): row['geo_mean_ons'] for _, row in geo_mean_df.iterrows()}

    def cost_fn(self):
        """
        Use the same cost function as simulated annealing for fair comparison
        """
        total_cost = 0
        for key, slot in self.train_scheduler.slots.items():
            day_type, hour, direction = key
            geo_mean_ons = self.geo_lookup.get((day_type, hour, direction), 0)
            trains = slot.current_trains
            
            # Constraint violations
            if trains < self.min_trains or trains > slot.max_slots:
                total_cost += self.penalties['constraint_violation']
                continue
            
            if trains > 0:
                # Load-based cost with same logic as simulated annealing
                load_per_train = geo_mean_ons / trains
                
                # Overload penalty 
                overload = max(0, load_per_train - self.train_capacity)
                total_cost += overload**2 * self.penalties['overload']
                
                # Underutilization penalty 
                underutil = max(0, self.train_capacity * 0.3 - load_per_train)  # Penalize if less than 30% full
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

    def optimize(self):
        """
        Run hill climbing optimization with multiple restarts.
        
        Returns:
            tuple: (best_solution, cost_progress)
        """
        # Initialize with same random strategy 
        for key, slot in self.train_scheduler.slots.items():
            min_possible = max(self.min_trains, int(60 / slot.max_frequency))
            max_possible = min(slot.max_slots, int(60 / slot.min_frequency))
            slot.current_trains = random.randint(min_possible, max_possible)
            
        current_cost = self.cost_fn()
        best_solution = {key: slot.current_trains for key, slot in self.train_scheduler.slots.items()}
        best_cost = current_cost
        print(f"HC Initial Cost: {current_cost:,.2f}")
        restart_count = 0
        max_restarts = 3
        no_improve_rounds = 0
        cost_progress = []  # Track cost at each iteration
        
        for restart in range(max_restarts):
            print(f"HC Restart {restart + 1}/{max_restarts}")
            for i in range(self.max_iterations // max_restarts):
                slot_key = random.choice(list(self.train_scheduler.slots.keys()))
                slot = self.train_scheduler.slots[slot_key]
                old_trains = slot.current_trains
                
                
                change = random.choice([-2, -1, 1, 2])  
                new_trains = slot.current_trains + change
                new_trains = max(self.min_trains, min(new_trains, slot.max_slots))
                slot.current_trains = new_trains
                
                new_cost = self.cost_fn()
                if new_cost < current_cost:
                    current_cost = new_cost
                    no_improve_rounds = 0
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_solution = {key: slot.current_trains for key, slot in self.train_scheduler.slots.items()}
                else:
                    slot.current_trains = old_trains
                    no_improve_rounds += 1
                
                # Track cost progress
                cost_progress.append(current_cost)
                
                if i % 100 == 0:
                    print(f"HC Iteration {i:4d}: Current Cost = {current_cost:,.2f}, Best Cost = {best_cost:,.2f}")
                if no_improve_rounds >= self.early_stopping_rounds:
                    print(f"Early stopping at iteration {i} due to no improvement.")
                    break
                    
            # Reinitialize for next restart 
            if restart < max_restarts - 1:
                for key, slot in self.train_scheduler.slots.items():
                    min_possible = max(self.min_trains, int(60 / slot.max_frequency))
                    max_possible = min(slot.max_slots, int(60 / slot.min_frequency))
                    slot.current_trains = random.randint(min_possible, max_possible)
                current_cost = self.cost_fn()
                
        print(f"HC Final Best Cost: {best_cost:,.2f}")
        
        # Set slots to best solution
        for key, trains in best_solution.items():
            self.train_scheduler.slots[key].current_trains = trains
            
        return best_solution, cost_progress 