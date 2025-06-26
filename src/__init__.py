"""
MBTA Optimal Schedule Finder

An AI-driven train scheduling optimization system using passenger flow data 
and GTFS (General Transit Feed Specification) data.

This package provides:
- Data preprocessing and validation
- Multiple optimization algorithms (Simulated Annealing, Hill Climbing, Genetic Algorithm)
- Visualization and reporting tools
- Configuration management

Author: Sujan DM, Ashwin G, Ajay
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Sujan DM, Ashwin G, Ajay"

# Import main components for easy access
from .main import main
from .optimization.simulated_annealing import TrainScheduler
from .visualization.visualize import ScheduleVisualizer

__all__ = [
    'main',
    'TrainScheduler', 
    'ScheduleVisualizer'
] 