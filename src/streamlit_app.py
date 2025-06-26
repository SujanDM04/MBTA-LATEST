import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
import glob


# Get the directory of the current script
SCRIPT_DIR = Path(__file__).parent

PROJECT_ROOT = (SCRIPT_DIR / '..').resolve()



ORANGE_LINE_STATIONS = [
    "Oak Grove", "Malden Center", "Wellington", "Assembly", "Sullivan Square", "Community College",
    "North Station", "Haymarket", "State", "Downtown Crossing", "Chinatown", "Tufts Medical Center",
    "Back Bay", "Massachusetts Avenue", "Ruggles", "Roxbury Crossing", "Jackson Square", "Stony Brook",
    "Green Street", "Forest Hills"
]


def get_day_type(date):
    """Determine day type based on the selected date"""
    day_of_week = date.weekday()  # Monday=0, Sunday=6
    if day_of_week == 5:  # Saturday
        return "saturday"
    elif day_of_week == 6:  # Sunday
        return "sunday"
    else:  # Monday-Friday
        return "weekday"


def load_schedule_for_day_type(day_type):
    """Load the appropriate schedule based on day type"""
    schedule_path = PROJECT_ROOT / "reports" / "train_schedule.csv"
    full_schedule_df = pd.read_csv(schedule_path)
    
    # Filter for the specific day type
    day_schedule = full_schedule_df[full_schedule_df['Day Type'] == day_type].copy()
    
    # Create a simplified schedule for the app 
    simplified_schedule = []
    for _, row in day_schedule.iterrows():
        simplified_schedule.append({
            'Station': row['Station'],
            'Departure Time': row['Departure Time']
        })
    
    return pd.DataFrame(simplified_schedule)


st.title("MBTA Orange Line Trip Planner")
st.write("Select your date, source, destination, and desired departure time. See the next 5 trains and estimated arrival times.")

# Date picker
selected_date = st.date_input(
    "Select Date",
    value=datetime.now().date(),
    min_value=datetime.now().date(),
    max_value=datetime.now().date() + timedelta(days=365)
)

# Determine day type based on selected date
day_type = get_day_type(selected_date)
day_name = selected_date.strftime("%A")

st.info(f"Selected date: {selected_date.strftime('%B %d, %Y')} ({day_name}) - Using {day_type} schedule")

# Load schedule for the selected day type
try:
    schedule_df = load_schedule_for_day_type(day_type)
    if schedule_df.empty:
        st.error(f"No schedule available for {day_type}. Please check the schedule files.")
        st.stop()
except Exception as e:
    st.error(f"Error loading schedule: {e}")
    st.stop()

# Display schedule info
schedule_path = PROJECT_ROOT / "reports" / "train_schedule.csv"
last_modified = datetime.fromtimestamp(schedule_path.stat().st_mtime)
st.info(f"Schedule last updated: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}")

lookup_df = pd.read_csv(PROJECT_ROOT / "reports" / "terminal_to_station_times.csv")
lookup_df.set_index('Station', inplace=True)


def get_terminal_and_direction(src, dst):
    src_idx = ORANGE_LINE_STATIONS.index(src)
    dst_idx = ORANGE_LINE_STATIONS.index(dst)
    if src_idx < dst_idx:
        terminal = "Oak Grove"
        direction = 0
    else:
        terminal = "Forest Hills"
        direction = 1
    return terminal, direction


src = st.selectbox("Source Station", ORANGE_LINE_STATIONS)
dst = st.selectbox("Destination Station", ORANGE_LINE_STATIONS, index=len(ORANGE_LINE_STATIONS)-1)

if src == dst:
    st.warning("Source and destination must be different.")
    st.stop()

time_str = st.text_input("Enter desired departure time (HH:MM, 24hr)", value="06:00")
try:
    user_time = datetime.strptime(time_str, "%H:%M")
except ValueError:
    st.error("Invalid time format. Please use HH:MM (24hr).")
    st.stop()


terminal, direction = get_terminal_and_direction(src, dst)

# Display day type and terminal info
st.subheader(f"Schedule Information")
st.write(f"**Day Type:** {day_type.title()}")
st.write(f"**Terminal:** {terminal}")
st.write(f"**Direction:** {'Northbound' if direction == 0 else 'Southbound'}")

terminal_schedule = schedule_df[schedule_df['Station'] == terminal].copy()
terminal_schedule['DepTimeObj'] = terminal_schedule['Departure Time'].apply(lambda t: datetime.strptime(t, "%H:%M"))

results = []
for _, row in terminal_schedule.iterrows():
    dep_time = row['DepTimeObj']
    
    if terminal == "Forest Hills":
        src_offset = lookup_df.loc[src, 'From Forest Hills (min)']
        dst_offset = lookup_df.loc[dst, 'From Forest Hills (min)']
    else:
        src_offset = lookup_df.loc[src, 'From Oak Grove (min)']
        dst_offset = lookup_df.loc[dst, 'From Oak Grove (min)']
    eta_src = (dep_time + timedelta(minutes=src_offset)).strftime("%H:%M")
    eta_dst = (dep_time + timedelta(minutes=dst_offset)).strftime("%H:%M")
    
    if datetime.strptime(eta_src, "%H:%M") >= user_time:
        results.append({
            "Train Departure Time (Terminal)": row['Departure Time'],
            "ETA at Source": eta_src,
            "ETA at Destination": eta_dst
        })

if not results:
    st.info("No more trains arriving at this station after the selected time.")
else:
    st.subheader(f"Next 5 trains arriving at {src} (from {terminal}) and their ETA at {dst}")
    st.table(pd.DataFrame(results).head(5))
    
    # Additional schedule information
    st.subheader("Schedule Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Trains Today", len(schedule_df))
    
    with col2:
        forest_hills_trains = len(schedule_df[schedule_df['Station'] == 'Forest Hills'])
        st.metric("Forest Hills Departures", forest_hills_trains)
    
    with col3:
        oak_grove_trains = len(schedule_df[schedule_df['Station'] == 'Oak Grove'])
        st.metric("Oak Grove Departures", oak_grove_trains) 