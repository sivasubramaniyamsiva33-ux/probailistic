"""
Root data package re-export for convenience.
"""
from canteen_queue.data.generate_data import generate_canteen_dataset, clean_data, load_or_generate_data

__all__ = ["generate_canteen_dataset", "clean_data", "load_or_generate_data"]
