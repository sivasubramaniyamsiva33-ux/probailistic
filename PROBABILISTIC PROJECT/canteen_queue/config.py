"""
Configuration parameters for Smart Queue Prediction and Waiting-Time Assistant
College Canteen Decision Support System (Saveetha College Canteen Queue Project)
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# Random Seed for Reproducibility across all experiments
GLOBAL_RANDOM_SEED: int = 42

# Time & Operation Constants
START_HOUR: int = 8       # 08:00 AM
END_HOUR: int = 18        # 06:00 PM
OPERATING_HOURS: int = END_HOUR - START_HOUR  # 10 hours
SLOT_MINUTES: int = 5     # 5-minute aggregation slot
SLOTS_PER_HOUR: int = 60 // SLOT_MINUTES      # 12 slots/hr
SLOTS_PER_DAY: int = OPERATING_HOURS * SLOTS_PER_HOUR  # 120 slots/day

# Canteen Counters
COUNTERS: List[str] = ["Meals", "Snacks", "Beverages"]

# Simulation / Historical Data Period
NUM_DAYS: int = 60        # 60 operational days

# Queue States (Discretized Congestion Buckets)
# State 0: Empty (0-1 customers)
# State 1: Short (2-4 customers)
# State 2: Medium (5-8 customers)
# State 3: Long (9-14 customers)
# State 4: Overcrowded (15+ customers)
QUEUE_STATE_NAMES: List[str] = ["Empty", "Short", "Medium", "Long", "Overcrowded"]
QUEUE_STATE_BOUNDS: List[Tuple[int, int]] = [
    (0, 1),
    (2, 4),
    (5, 8),
    (9, 14),
    (15, 999),
]

# Baseline Service Parameters (Service rate mu in customers per minute)
# Mean service times: Meals = 75s (0.8/min), Snacks = 40s (1.5/min), Beverages = 25s (2.4/min)
DEFAULT_SERVICE_RATES: Dict[str, float] = {
    "Meals": 0.8,       # ~75 seconds per customer
    "Snacks": 1.5,      # ~40 seconds per customer
    "Beverages": 2.4,   # ~25 seconds per customer
}

# Default Active Servers (c) per counter
DEFAULT_SERVERS: Dict[str, int] = {
    "Meals": 2,
    "Snacks": 1,
    "Beverages": 1,
}

# Time-of-Day Bands for Arrival Modeling
TIME_BANDS: Dict[str, Tuple[int, int]] = {
    "Morning": (8, 11),       # 08:00 - 11:00
    "Lunch": (11, 14),        # 11:00 - 14:00 (Peak 12:30 - 14:00)
    "Afternoon": (14, 18),    # 14:00 - 18:00 (Break spike 15:30)
}

# Peak Hours Definitions
LUNCH_PEAK_START: float = 12.5    # 12:30 PM
LUNCH_PEAK_END: float = 14.0      # 02:00 PM
MORNING_BREAK: float = 10.5       # 10:30 AM
AFTERNOON_BREAK: float = 15.5     # 03:30 PM

# Traffic Light Waiting Time Thresholds (in minutes)
TRAFFIC_LIGHT_GREEN_MAX: float = 5.0    # < 5 minutes -> Green
TRAFFIC_LIGHT_AMBER_MAX: float = 12.0   # 5 - 12 minutes -> Amber; > 12 -> Red

# File and Directory Paths
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_PATH: Path = DATA_DIR / "canteen_records_raw.csv"
CLEAN_DATA_PATH: Path = DATA_DIR / "canteen_records_clean.csv"
REPORTS_DIR: Path = BASE_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
