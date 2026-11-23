"""
Synthetic Canteen Queue Data Generator & Cleaning Pipeline
Saveetha College Canteen Queue Assistant
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Support both relative and absolute execution paths
try:
    from canteen_queue.config import (
        CLEAN_DATA_PATH,
        COUNTERS,
        DATA_DIR,
        DEFAULT_SERVERS,
        DEFAULT_SERVICE_RATES,
        END_HOUR,
        GLOBAL_RANDOM_SEED,
        NUM_DAYS,
        RAW_DATA_PATH,
        SLOT_MINUTES,
        SLOTS_PER_DAY,
        START_HOUR,
    )
except ImportError:
    # If executed directly as python data/generate_data.py
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config import (
        CLEAN_DATA_PATH,
        COUNTERS,
        DATA_DIR,
        DEFAULT_SERVERS,
        DEFAULT_SERVICE_RATES,
        END_HOUR,
        GLOBAL_RANDOM_SEED,
        NUM_DAYS,
        RAW_DATA_PATH,
        SLOT_MINUTES,
        SLOTS_PER_DAY,
        START_HOUR,
    )


def compute_baseline_lambda(
    counter: str,
    time_decimal: float,
    day_name: str,
    day_idx: int,
) -> float:
    """
    Computes expected Poisson arrival rate (arrivals per 5-min slot)
    incorporating diurnal schedule peaks, college breaks, day-of-week,
    and exam week surges.
    """
    # 1. Base counter rate profiles across the day
    if counter == "Meals":
        # Meals peak strongly at lunch (12:30 - 14:00), minimal in early morning / late afternoon
        base = 0.8
        # Morning breakfast curve (08:30 - 10:00)
        breakfast_bump = 3.5 * np.exp(-0.5 * ((time_decimal - 9.0) / 0.6) ** 2)
        # Lunch peak (12:30 - 14:00 centered at 13.15)
        lunch_bump = 14.0 * np.exp(-0.5 * ((time_decimal - 13.2) / 0.8) ** 2)
        # Early evening snack/tea (16:30 - 17:30)
        evening_bump = 2.0 * np.exp(-0.5 * ((time_decimal - 17.0) / 0.6) ** 2)
        rate = base + breakfast_bump + lunch_bump + evening_bump

    elif counter == "Snacks":
        # Snacks peak during class breaks (10:30 and 15:30) and steady lunch
        base = 2.5
        # Morning break (10:30 AM)
        break1_bump = 8.5 * np.exp(-0.5 * ((time_decimal - 10.5) / 0.45) ** 2)
        # Lunch snacks (12:30 - 13:45)
        lunch_bump = 6.0 * np.exp(-0.5 * ((time_decimal - 13.0) / 0.9) ** 2)
        # Afternoon break (15:30 PM)
        break2_bump = 9.5 * np.exp(-0.5 * ((time_decimal - 15.5) / 0.45) ** 2)
        rate = base + break1_bump + lunch_bump + break2_bump

    elif counter == "Beverages":
        # Beverages: tea/coffee spikes during morning entry, breaks, and exam week
        base = 3.0
        # Morning arrival surge (08:15 - 09:15)
        morning_cup = 6.0 * np.exp(-0.5 * ((time_decimal - 8.6) / 0.5) ** 2)
        # Morning break (10:30 AM)
        break1_cup = 10.0 * np.exp(-0.5 * ((time_decimal - 10.5) / 0.4) ** 2)
        # Post-lunch drink (13:15 - 14:15)
        lunch_cup = 8.5 * np.exp(-0.5 * ((time_decimal - 13.5) / 0.6) ** 2)
        # Afternoon tea break (15:30 PM)
        break2_cup = 11.5 * np.exp(-0.5 * ((time_decimal - 15.5) / 0.45) ** 2)
        rate = base + morning_cup + break1_cup + lunch_cup + break2_cup
    else:
        rate = 4.0

    # 2. Day-of-week factor: Friday has lower attendance; Saturday is half/moderate
    if day_name == "Friday":
        rate *= 0.80  # 20% lower demand on Fridays
    elif day_name == "Saturday":
        rate *= 0.65  # Saturday short academic schedule

    # 3. Exam-week surge: Days 44 to 49 (6-day examination period)
    # Students consume significantly more beverages and quick snacks
    if 44 <= day_idx <= 49:
        if counter == "Beverages":
            rate *= 1.70  # +70% beverage demand (coffee/tea/energy)
        elif counter == "Snacks":
            rate *= 1.35  # +35% fast finger-food demand
        elif counter == "Meals":
            rate *= 0.85  # Meals slight drop as students eat quick bites

    return float(np.maximum(0.2, rate))


def generate_canteen_dataset(
    num_days: int = NUM_DAYS,
    seed: int = GLOBAL_RANDOM_SEED,
    inject_anomalies: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates 60 days of synthetic records for 3 counters across 8:00 - 18:00.
    Returns:
        (raw_df, clean_df)
    """
    np.random.seed(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    base_date = datetime(2026, 9, 1, START_HOUR, 0, 0)
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

    for day_idx in range(num_days):
        day_of_week_idx = day_idx % 6  # 6-day academic week (Mon-Sat)
        day_name = day_names[day_of_week_idx]
        current_date = base_date + timedelta(days=day_idx)

        # Dynamic state tracking per counter for queue evolution across slots
        current_queue_len = {c: 0 for c in COUNTERS}

        for slot_idx in range(SLOTS_PER_DAY):
            slot_offset_minutes = slot_idx * SLOT_MINUTES
            record_dt = current_date + timedelta(minutes=slot_offset_minutes)
            time_decimal = START_HOUR + (slot_offset_minutes / 60.0)
            time_slot_str = record_dt.strftime("%H:%M")

            for counter in COUNTERS:
                # 1. Arrival process
                expected_lambda = compute_baseline_lambda(
                    counter=counter,
                    time_decimal=time_decimal,
                    day_name=day_name,
                    day_idx=day_idx,
                )
                # Poisson arrivals with slight Gamma environmental overdispersion
                gamma_noise = np.random.gamma(shape=10.0, scale=0.1)
                actual_lambda = expected_lambda * gamma_noise
                arrivals = int(np.random.poisson(lam=actual_lambda))

                # 2. Service process
                mu_per_minute = DEFAULT_SERVICE_RATES[counter]
                mu_slot = mu_per_minute * SLOT_MINUTES  # customers per server per 5 min
                c_servers = DEFAULT_SERVERS[counter]
                total_capacity = c_servers * mu_slot

                # Customer service time: Gamma distribution (Erlang k=2 or k=3 shape)
                # Mean in seconds = (60 / mu_per_minute)
                mean_service_sec = 60.0 / mu_per_minute
                shape_k = 2.5
                scale_theta = mean_service_sec / shape_k
                sampled_service_sec = float(np.random.gamma(shape=shape_k, scale=scale_theta))

                # 3. Queue evolution
                queue_at_arrival = current_queue_len[counter]
                # Potential departures during this 5-minute interval
                actual_service_rate_per_min = 60.0 / np.maximum(5.0, sampled_service_sec)
                departures_capacity = np.random.poisson(
                    lam=actual_service_rate_per_min * c_servers * SLOT_MINUTES
                )
                # Customers served are bounded by (current queue + arrivals)
                departures = min(queue_at_arrival + arrivals, departures_capacity)

                # Update queue for next slot
                next_queue = max(0, queue_at_arrival + arrivals - departures)
                current_queue_len[counter] = next_queue

                # 4. Realistic observed wait time (minutes)
                # If newcomer arrives when queue length is Q, wait in queue involves Q / (c * mu)
                # plus stochastic service time
                service_time_min = sampled_service_sec / 60.0
                effective_servers = max(1, c_servers)
                queue_delay_min = (queue_at_arrival / (effective_servers * mu_per_minute)) + np.random.exponential(
                    scale=max(0.1, 0.4 / (effective_servers * mu_per_minute))
                )
                observed_wait = float(np.maximum(0.2, queue_delay_min + service_time_min))

                records.append(
                    {
                        "timestamp": record_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "day_index": day_idx,
                        "day_of_week": day_name,
                        "time_slot": time_slot_str,
                        "slot_index": slot_idx,
                        "counter": counter,
                        "arrival_count": arrivals,
                        "service_time_seconds": round(sampled_service_sec, 2),
                        "queue_length_at_arrival": int(queue_at_arrival),
                        "observed_wait_time": round(observed_wait, 2),
                    }
                )

    raw_df = pd.DataFrame(records)

    # 5. Inject 5% anomalies/missing values into raw_df if requested
    if inject_anomalies:
        raw_df = inject_synthetic_anomalies(raw_df, seed=seed)

    # 6. Apply data cleaning pipeline
    clean_df = clean_data(raw_df)

    # Save to disk
    raw_df.to_csv(RAW_DATA_PATH, index=False)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)

    return raw_df, clean_df


def inject_synthetic_anomalies(df: pd.DataFrame, seed: int = GLOBAL_RANDOM_SEED) -> pd.DataFrame:
    """
    Injects realistic 5% anomalies into the dataset:
    - 2.0% Missing / NaN values
    - 1.5% Negative corrupted sensor values
    - 1.5% Extreme sensor blowout spikes (e.g. 500 arrivals)
    """
    np.random.seed(seed + 101)
    df_noisy = df.copy()
    n_rows = len(df_noisy)

    # Total 5% anomaly rows
    anomaly_indices = np.random.choice(n_rows, size=int(0.05 * n_rows), replace=False)
    split1 = int(0.40 * len(anomaly_indices))  # 2.0% NaNs
    split2 = int(0.70 * len(anomaly_indices))  # 1.5% Negatives
    # remainder: 1.5% Extreme spikes

    nan_idx = anomaly_indices[:split1]
    neg_idx = anomaly_indices[split1:split2]
    spike_idx = anomaly_indices[split2:]

    # Inject NaNs
    for idx in nan_idx:
        col = np.random.choice(
            ["arrival_count", "service_time_seconds", "queue_length_at_arrival", "observed_wait_time"]
        )
        df_noisy.at[idx, col] = np.nan

    # Inject Negatives
    for idx in neg_idx:
        col = np.random.choice(["arrival_count", "queue_length_at_arrival", "observed_wait_time"])
        val = df_noisy.at[idx, col]
        if pd.notna(val):
            df_noisy.at[idx, col] = -abs(float(val)) - np.random.randint(1, 15)

    # Inject Extreme Spikes
    for idx in spike_idx:
        col = np.random.choice(["arrival_count", "queue_length_at_arrival", "observed_wait_time"])
        if col == "arrival_count":
            df_noisy.at[idx, col] = np.random.randint(250, 999)
        elif col == "queue_length_at_arrival":
            df_noisy.at[idx, col] = np.random.randint(180, 500)
        else:
            df_noisy.at[idx, col] = round(float(np.random.uniform(90.0, 300.0)), 2)

    return df_noisy


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans synthetic canteen records:
    1. Replaces negative and extreme outlier values with NaN
    2. Uses grouped median imputation (by counter, time_slot, day_of_week)
    3. Guarantees non-negative integer arrival counts and queue lengths
    4. Guarantees realistic positive service times and wait times
    """
    clean_df = df.copy()

    # 1. Mark negative corrupted values as NaN
    numeric_cols = [
        "arrival_count",
        "service_time_seconds",
        "queue_length_at_arrival",
        "observed_wait_time",
    ]
    for col in numeric_cols:
        clean_df.loc[clean_df[col] < 0, col] = np.nan

    # 2. Mark absurd extreme spikes as NaN
    clean_df.loc[clean_df["arrival_count"] > 60, "arrival_count"] = np.nan
    clean_df.loc[clean_df["queue_length_at_arrival"] > 80, "queue_length_at_arrival"] = np.nan
    clean_df.loc[clean_df["observed_wait_time"] > 45.0, "observed_wait_time"] = np.nan
    clean_df.loc[clean_df["service_time_seconds"] > 240.0, "service_time_seconds"] = np.nan
    clean_df.loc[clean_df["service_time_seconds"] < 5.0, "service_time_seconds"] = np.nan

    # 3. Grouped median imputation by (counter, time_slot, day_of_week)
    for col in numeric_cols:
        group_medians = clean_df.groupby(["counter", "time_slot", "day_of_week"])[col].transform("median")
        clean_df[col] = clean_df[col].fillna(group_medians)
        # Fallback to counter median if still NaN
        counter_medians = clean_df.groupby("counter")[col].transform("median")
        clean_df[col] = clean_df[col].fillna(counter_medians)
        # Final global median fallback
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    # 4. Cast counts to integers and round values
    clean_df["arrival_count"] = clean_df["arrival_count"].round().astype(int)
    clean_df["queue_length_at_arrival"] = clean_df["queue_length_at_arrival"].round().astype(int)
    clean_df["service_time_seconds"] = clean_df["service_time_seconds"].round(2)
    clean_df["observed_wait_time"] = clean_df["observed_wait_time"].round(2)

    return clean_df


def load_or_generate_data(force_regenerate: bool = False) -> pd.DataFrame:
    """
    Convenience helper: returns cleaned dataset. If clean CSV does not exist
    or force_regenerate is True, generates from scratch.
    """
    if CLEAN_DATA_PATH.exists() and not force_regenerate:
        return pd.read_csv(CLEAN_DATA_PATH)
    _, clean_df = generate_canteen_dataset(num_days=NUM_DAYS, inject_anomalies=True)
    return clean_df


if __name__ == "__main__":
    print(f"[Saveetha College Canteen Queue] Generating {NUM_DAYS} days of synthetic records...")
    raw_df, clean_df = generate_canteen_dataset(num_days=NUM_DAYS, inject_anomalies=True)
    print(f"Generated raw records: {len(raw_df)} rows saved to {RAW_DATA_PATH}")
    print(f"Cleaned records: {len(clean_df)} rows saved to {CLEAN_DATA_PATH}")
    print("\nSummary of Cleaned Data:")
    print(clean_df.groupby("counter")[["arrival_count", "queue_length_at_arrival", "observed_wait_time"]].mean())
