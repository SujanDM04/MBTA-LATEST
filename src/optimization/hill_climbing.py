import numpy as np
import random

class HillClimbingTrainScheduler:
    def __init__(self, train_scheduler, geo_mean_df, train_capacity=1000, min_trains=3, max_iterations=1000, early_stopping_rounds=200):
        """
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
        # Penalty configuration
        self.penalties = {
            'constraint_violation': 1e6,
            'overload': 1000,
            'underutil': 200,
            'frequency': 500,
            'crowding': 2000,
            'no_trains': 1e8
        }
        # Preprocess geo_mean_df into a dict for fast lookup
        self.geo_lookup = {(row['day_type_name'], row['hour'], row['direction_id']): row['geo_mean_ons'] for _, row in geo_mean_df.iterrows()}

    def cost_fn(self):
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
                load_per_train = geo_mean_ons / trains
                overload = max(0, load_per_train - self.train_capacity)
                total_cost += overload * self.penalties['overload']
                underutil = max(0, self.train_capacity * 0.5 - load_per_train)
                total_cost += underutil * self.penalties['underutil']
                frequency = 60 / trains
                if frequency < slot.min_frequency:
                    total_cost += (slot.min_frequency - frequency) * self.penalties['frequency']
                elif frequency > slot.max_frequency:
                    total_cost += (frequency - slot.max_frequency) * self.penalties['frequency']
                if frequency < 5:
                    total_cost += (5 - frequency) * self.penalties['crowding']
            else:
                if geo_mean_ons > 0:
                    total_cost += self.penalties['no_trains']
        return total_cost

    def optimize(self):
        for key, slot in self.train_scheduler.slots.items():
            if random.random() < 0.3:
                slot.current_trains = self.min_trains
            elif random.random() < 0.5:
                slot.current_trains = slot.max_slots
            else:
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
                changes = [-1, 1]
                change = random.choice(changes)
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
            if restart < max_restarts - 1:
                for key, slot in self.train_scheduler.slots.items():
                    min_possible = max(self.min_trains, int(60 / slot.max_frequency))
                    max_possible = min(slot.max_slots, int(60 / slot.min_frequency))
                    slot.current_trains = random.randint(min_possible, max_possible)
                current_cost = self.cost_fn()
        print(f"HC Final Best Cost: {best_cost:,.2f}")
        for key, trains in best_solution.items():
            self.train_scheduler.slots[key].current_trains = trains
        return best_solution, cost_progress 