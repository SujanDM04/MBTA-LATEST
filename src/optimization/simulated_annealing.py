import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import random
from optimization.cost import CostFunction

@dataclass
class TimeSlot:
    day_type: str
    time_period: str
    max_slots: int
    current_trains: int
    demand: float
    min_frequency: int = 15
    max_frequency: int = 30

class SimulatedAnnealingTrainScheduler:
    def __init__(self, 
                 passenger_data: pd.DataFrame,
                 time_slots: pd.DataFrame,
                 train_capacity: int = 1000,
                 min_trains: int = 3,
                 min_frequency: int = 15,
                 max_frequency: int = 30):
        self.passenger_data = passenger_data
        self.time_slots = time_slots
        self.train_capacity = train_capacity
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.min_trains = min_trains
        self.slots: Dict[tuple, TimeSlot] = {}
        self.penalties = {
            'constraint_violation': 1e6,
            'overload': 10,
            'underutil': 100,
            'wait_time': 0.1,
            'frequency': 100,
            'no_trains': 1e8
        }
        self.geo_lookup = None
        
    def initialize_slots(self, onboard_demand_df=None):
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
            demand = None
            if onboard_lookup is not None:
                demand = onboard_lookup.get(key, 0)
            if demand is None:
                demand = row['total_ons'] + row['total_offs']
            self.slots[key] = TimeSlot(
                day_type=row['day_type_name'],
                time_period=row['hour'],
                max_slots=max_slots,
                current_trains=default_trains,
                demand=demand,
                min_frequency=self.min_frequency,
                max_frequency=self.max_frequency
            )
    
    def calculate_cost(self) -> float:
        total_cost = 0
        for slot in self.slots.values():
            if slot.current_trains > 0:
                frequency = 60 / slot.current_trains
                wait_time = frequency / 2
                load_per_train = slot.demand / slot.current_trains
                wait_time_cost = wait_time * slot.demand * self.penalties['wait_time']
                capacity_cost = (load_per_train - self.train_capacity) ** 2
                if frequency < slot.min_frequency:
                    frequency_cost = (slot.min_frequency - frequency) * self.penalties['frequency']
                elif frequency > slot.max_frequency:
                    frequency_cost = (frequency - slot.max_frequency) * self.penalties['frequency']
                else:
                    frequency_cost = 0
                total_cost += wait_time_cost + capacity_cost + frequency_cost
        return total_cost
    
    def optimize_frequency(self, geo_mean_df, max_iterations: int = 1000) -> dict:
        self.initialize_slots()
        self.geo_lookup = {(row['day_type_name'], row['hour'], row['direction_id']): row['geo_mean_ons'] for _, row in geo_mean_df.iterrows()}
        for key, slot in self.slots.items():
            min_possible = max(self.min_trains, int(60 / slot.max_frequency))
            max_possible = min(slot.max_slots, int(60 / slot.min_frequency))
            slot.current_trains = random.randint(min_possible, max_possible)
        current_cost = CostFunction(self).cost_fn()
        best_solution = {key: slot.current_trains for key, slot in self.slots.items()}
        best_cost = current_cost
        temperature = 1000.0
        cooling_rate = 0.995
        cost_progress = []
        for i in range(max_iterations):
            num_changes = random.randint(1, 3)
            old_trains = {}
            for _ in range(num_changes):
                slot_key = random.choice(list(self.slots.keys()))
                slot = self.slots[slot_key]
                old_trains[slot_key] = slot.current_trains
                change = random.choice([-2, -1, 1, 2])
                new_trains = slot.current_trains + change
                new_trains = max(self.min_trains, min(new_trains, slot.max_slots))
                slot.current_trains = new_trains
            new_cost = CostFunction(self).cost_fn()
            cost_diff = new_cost - current_cost
            if cost_diff < 0 or random.random() < np.exp(-cost_diff / temperature):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = {key: slot.current_trains for key, slot in self.slots.items()}
            else:
                for slot_key, old_train_count in old_trains.items():
                    self.slots[slot_key].current_trains = old_train_count
            cost_progress.append(current_cost)
            temperature *= cooling_rate
        for key, trains in best_solution.items():
            self.slots[key].current_trains = trains
        for key, slot in self.slots.items():
            frequency = 60 / slot.current_trains if slot.current_trains > 0 else float('inf')
            if frequency < slot.min_frequency:
                slot.current_trains = max(slot.current_trains - 1, self.min_trains)
            elif frequency > slot.max_frequency:
                slot.current_trains = min(slot.current_trains + 1, slot.max_slots)
        return best_solution, cost_progress

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


