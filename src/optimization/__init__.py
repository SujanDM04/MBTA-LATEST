"""
Optimization Algorithms Package

This package contains various optimization algorithms for train scheduling:
- Simulated Annealing: Global optimization with temperature-based acceptance
- Hill Climbing: Local search with greedy improvement
- Genetic Algorithm: Population-based evolutionary optimization

Each algorithm implements the same interface for easy comparison and switching.
"""

from .simulated_annealing import TrainScheduler as SimulatedAnnealingScheduler
from .hill_climbing import HillClimbingTrainScheduler
from .genetic_algorithm_experiment.genetic_algorithm import GeneticAlgorithmScheduler

__all__ = [
    'SimulatedAnnealingScheduler',
    'HillClimbingTrainScheduler', 
    'GeneticAlgorithmScheduler'
] 