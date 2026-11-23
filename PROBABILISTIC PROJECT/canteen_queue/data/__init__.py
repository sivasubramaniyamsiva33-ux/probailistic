"""
Data generation and cleaning package for canteen records.
"""
from .generate_data import generate_canteen_dataset, clean_data, load_or_generate_data

__all__ = ["generate_canteen_dataset", "clean_data", "load_or_generate_data"]
