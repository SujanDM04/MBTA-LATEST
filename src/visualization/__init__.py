"""
Visualization Package

This package provides comprehensive visualization and reporting tools:
- Demand distribution plots
- Train allocation visualizations
- Load distribution analysis
- Comparative analysis between algorithms
- Schedule generation and export
"""

from .visualize import ScheduleVisualizer, generate_full_eta_table, build_terminal_to_station_lookup

__all__ = [
    'ScheduleVisualizer',
    'generate_full_eta_table',
    'build_terminal_to_station_lookup'
] 