"""
Root-level script runner for data/generate_data.py
"""
import sys
from pathlib import Path

# Add current workspace root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from canteen_queue.data.generate_data import generate_canteen_dataset, NUM_DAYS, RAW_DATA_PATH, CLEAN_DATA_PATH

if __name__ == "__main__":
    print(f"[Saveetha College Canteen Queue] Generating {NUM_DAYS} days of synthetic records...")
    raw_df, clean_df = generate_canteen_dataset(num_days=NUM_DAYS, inject_anomalies=True)
    print(f"Generated raw records: {len(raw_df)} rows saved to {RAW_DATA_PATH}")
    print(f"Cleaned records: {len(clean_df)} rows saved to {CLEAN_DATA_PATH}")
    print("\nSummary of Cleaned Data:")
    print(clean_df.groupby("counter")[["arrival_count", "queue_length_at_arrival", "observed_wait_time"]].mean())
