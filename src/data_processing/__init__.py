"""
Data Processing Package

This package handles all data preprocessing, validation, and transformation:
- Passenger flow data processing
- GTFS data parsing and validation
- Data cleaning and normalization
- Feature engineering for optimization
"""

from .preprocess import DataPreprocessor

__all__ = [
    'DataPreprocessor'
] 