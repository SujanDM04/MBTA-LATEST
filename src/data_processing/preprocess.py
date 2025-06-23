import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

class DataPreprocessor:
    def __init__(self, passenger_data_path: str, gtfs_data_path: str):
        """
        Initialize the data preprocessor.
        
        Args:
            passenger_data_path: Path to passenger flow data
            gtfs_data_path: Path to GTFS data directory
        """
        self.passenger_data_path = Path(passenger_data_path)
        self.gtfs_data_path = Path(gtfs_data_path)
        
    def load_passenger_data(self) -> pd.DataFrame:
        """
        Load and preprocess passenger flow data, mapping columns from the user's CSV format.
        
        Returns:
            DataFrame with processed passenger flow data
        """
        df = pd.read_csv(self.passenger_data_path)
        df.columns = df.columns.str.strip()  # Strip whitespace from column names
        # Map/rename columns to required names
        df = df.rename(columns={
            'stop_name': 'stop_name',
            'parent_station': 'parent_station',
            'day_type_name': 'day_type_name',
            'time_period_name': 'time_period_name',
            'total_ons': 'total_ons',
            'total_offs': 'total_offs',
            'direction_id': 'direction_id'
        })
        df['parent_station'] = df['stop_name']
        if 'direction_id' not in df.columns:
            df['direction_id'] = 0
        # Keep only required columns
        df = df[['stop_name', 'parent_station', 'day_type_name', 'time_period_name', 'direction_id', 'total_ons', 'total_offs']]
        # Clean and preprocess data
        df['day_type_name'] = df['day_type_name'].astype(str).str.lower()
        df['time_period_name'] = df['time_period_name'].astype(str).str.upper()
        return df
    
    def load_gtfs_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load GTFS data files from the data/gtfs/ folder.
        
        Returns:
            Tuple of (stop_times, trips, calendar) DataFrames
        """
        stop_times = pd.read_csv(self.gtfs_data_path / 'stop_times.txt')
        trips = pd.read_csv(self.gtfs_data_path / 'trips.txt')
        calendar = pd.read_csv(self.gtfs_data_path / 'calendar.txt')
        
        return stop_times, trips, calendar
    
    def calculate_time_slots(self, stop_times: pd.DataFrame, passenger_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate available time slots from GTFS data and map to day_type and time_period.
        
        Args:
            stop_times: DataFrame containing stop times
            passenger_data: DataFrame containing passenger data (for day_type/time_period mapping)
            
        Returns:
            DataFrame with time slot information, including day_type and time_period
        """
        # Fix 24:xx:xx times to 00:xx:xx (next day)
        stop_times['arrival_time'] = stop_times['arrival_time'].astype(str).str.replace(r'^24:(..:..)$', r'00:\1', regex=True)
        stop_times['arrival_time'] = pd.to_datetime(stop_times['arrival_time'], format='%H:%M:%S', errors='coerce')
        stop_times = stop_times.dropna(subset=['arrival_time'])
        stop_times['time_slot'] = stop_times['arrival_time'].dt.floor('15min')
        
        # Count trips per time slot
        slots = stop_times.groupby('time_slot').size().reset_index(name='trip_count')
        
        # Map each time_slot to a time_period using passenger_data's unique time_periods
        # We'll assume time_periods are defined by time ranges (e.g., AM_RUSH = 07:00-09:00)
        # For now, we'll just assign all slots to all day_type/time_period combos (for demo)
        unique_periods = passenger_data[['day_type_name', 'time_period_name']].drop_duplicates()
        unique_periods['key'] = 1
        slots['key'] = 1
        slots = slots.merge(unique_periods, on='key').drop('key', axis=1)
        
        # Rearrange columns
        slots = slots[['day_type_name', 'time_period_name', 'time_slot', 'trip_count']]
        
        return slots
    
    def process_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Process passenger data and simulate train slots per hour and direction (ignore GTFS).
        Returns:
            Tuple of (hourly passenger data, hourly slots data, hourly geometric mean demand)
        """
        # Load and process passenger data
        passenger_data = self.load_passenger_data()

        # Standard MBTA mapping: time_period_name -> hour ranges (24-hour, end exclusive)
        period_to_hours = {
            'VERY_EARLY_MORNING': list(range(4, 6)),
            'EARLY_AM': list(range(6, 7)),
            'AM_PEAK': list(range(7, 9)),
            'MIDDAY_SCHOOL': list(range(9, 14)),
            'MIDDAY_BASE': list(range(9, 16)),
            'PM_PEAK': list(range(16, 18)),
            'EVENING': list(range(18, 21)),
            'LATE_EVENING': list(range(21, 24)),
            'NIGHT': list(range(0, 4)),
            'OFF_PEAK': list(range(4, 24)),  # fallback for any unmapped period
        }

        # Expand passenger data to hourly granularity, including direction
        hourly_rows = []
        for _, row in passenger_data.iterrows():
            period = row['time_period_name']
            hours = period_to_hours.get(period, [])
            if not hours:
                continue  # skip if period not mapped
            ons_per_hour = row['total_ons'] / len(hours)
            offs_per_hour = row['total_offs'] / len(hours)
            for hour in hours:
                hourly_rows.append({
                    'stop_name': row['stop_name'],
                    'parent_station': row['parent_station'],
                    'day_type_name': row['day_type_name'],
                    'direction_id': row.get('direction_id', 0),
                    'hour': hour,
                    'total_ons': ons_per_hour,
                    'total_offs': offs_per_hour
                })
        hourly_passenger_df = pd.DataFrame(hourly_rows)

        # Compute geometric mean of total_ons for each (day_type_name, hour, direction_id)
        def geo_mean(x):
            return np.exp(np.log(x[x > 0]).mean()) if np.any(x > 0) else 0
        geo_mean_df = hourly_passenger_df.groupby(['day_type_name', 'hour', 'direction_id'])['total_ons'].agg(geo_mean).reset_index()
        geo_mean_df = geo_mean_df.rename(columns={'total_ons': 'geo_mean_ons'})

        # Compute max_onboard for each slot and merge into hourly_passenger_df
        onboard_demand_df = self.compute_onboard_demand(hourly_passenger_df)
        hourly_passenger_df = hourly_passenger_df.merge(
            onboard_demand_df,
            on=['day_type_name', 'hour', 'direction_id'],
            how='left'
        )
        # Use hardcoded service days for normalization
        service_days_lookup = {
            'weekday': 77,
            'saturday': 12,
            'sunday': 8
        }
        hourly_passenger_df['adjusted_max_onboard'] = hourly_passenger_df.apply(
            lambda row: row['max_onboard'] / service_days_lookup.get(row['day_type_name'], 1), axis=1
        )

        # Simulate train slots: 10 slots/hour, default 6 trains/hour, for each hour and direction
        slot_rows = []
        for day_type in hourly_passenger_df['day_type_name'].unique():
            for direction in hourly_passenger_df['direction_id'].unique():
                for hour in range(4, 24):
                    slot_rows.append({
                        'day_type_name': day_type,
                        'direction_id': direction,
                        'hour': hour,
                        'max_slots': 10,
                        'default_trains': 6
                    })
                slot_rows.append({  # 1:00 AM next day
                    'day_type_name': day_type,
                    'direction_id': direction,
                    'hour': 1,
                    'max_slots': 10,
                    'default_trains': 6
                })
        slots_df = pd.DataFrame(slot_rows)

        return hourly_passenger_df, slots_df, geo_mean_df

    def compute_onboard_demand(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the maximum onboard passenger load for each (day_type_name, hour, direction_id) group.
        For each group, sort by stop_id (or stop_name if stop_id is missing), simulate onboard load as:
            onboard[t] = onboard[t-1] + total_ons[t] - total_offs[t]
        Ensures onboard is never negative. If group is missing, max_onboard=0.
        Returns a DataFrame with columns: day_type_name, hour, direction_id, max_onboard
        """
        onboard_records = []
        # Group by day_type_name, hour, direction_id (changed from time_period_name)
        group_cols = ['day_type_name', 'hour', 'direction_id']
        for group, group_df in df.groupby(group_cols):
            # Sort by stop_id if present, else stop_name
            if 'stop_id' in group_df.columns:
                group_df = group_df.sort_values('stop_id')
            else:
                group_df = group_df.sort_values('stop_name')
            onboard = 0
            onboard_list = []
            # Simulate rolling onboard load
            for _, row in group_df.iterrows():
                onboard = max(0, onboard + row['total_ons'] - row['total_offs'])
                onboard_list.append(onboard)
            max_onboard = max(onboard_list) if onboard_list else 0
            onboard_records.append({
                'day_type_name': group[0],
                'hour': group[1],  # changed from time_period_name
                'direction_id': group[2],
                'max_onboard': max_onboard
            })
        onboard_df = pd.DataFrame(onboard_records)
        # Fill missing groups with max_onboard=0
        all_groups = df[group_cols].drop_duplicates()
        onboard_df = all_groups.merge(onboard_df, on=group_cols, how='left').fillna({'max_onboard': 0})
        onboard_df['max_onboard'] = onboard_df['max_onboard'].astype(int)
        return onboard_df

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(
        passenger_data_path="data/passenger_flow/passenger_Data.csv",
        gtfs_data_path="data/gtfs"
    )
    
    passenger_data, time_slots, geo_mean_demand = preprocessor.process_data()
    print("Data processing completed successfully!") 