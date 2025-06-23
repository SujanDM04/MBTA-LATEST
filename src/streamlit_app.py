import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
import glob

# --- Robust Path Loading ---
# Get the directory of the current script
SCRIPT_DIR = Path(__file__).parent
# Explicitly go up one directory to find the project root
PROJECT_ROOT = (SCRIPT_DIR / '..').resolve()
# ---

# Orange Line stations in order
ORANGE_LINE_STATIONS = [
    "Oak Grove", "Malden Center", "Wellington", "Assembly", "Sullivan Square", "Community College",
    "North Station", "Haymarket", "State", "Downtown Crossing", "Chinatown", "Tufts Medical Center",
    "Back Bay", "Massachusetts Avenue", "Ruggles", "Roxbury Crossing", "Jackson Square", "Stony Brook",
    "Green Street", "Forest Hills"
]

# Load schedule (Simulated Annealing)
schedule_path = PROJECT_ROOT / "reports" / "simple_schedule_sa.csv"
schedule_df = pd.read_csv(schedule_path)

# Show last modified time of the schedule file
last_modified = datetime.fromtimestamp(schedule_path.stat().st_mtime)
st.info(f"Schedule last updated: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}")

# Load terminal-to-station lookup table
lookup_df = pd.read_csv(PROJECT_ROOT / "reports" / "terminal_to_station_times.csv")
lookup_df.set_index('Station', inplace=True)

# Helper to get terminal and direction for a source/destination pair
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

# Streamlit UI
st.title("MBTA Orange Line Trip Planner")
st.write("Select your source, destination, and desired departure time. See the next 5 trains and estimated arrival times.")

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

# Determine terminal and direction
terminal, direction = get_terminal_and_direction(src, dst)

# Get all departures from the relevant terminal
terminal_schedule = schedule_df[schedule_df['Station'] == terminal].copy()
terminal_schedule['DepTimeObj'] = terminal_schedule['Departure Time'].apply(lambda t: datetime.strptime(t, "%H:%M"))

results = []
for _, row in terminal_schedule.iterrows():
    dep_time = row['DepTimeObj']
    # Lookup cumulative time from terminal to source and destination
    if terminal == "Forest Hills":
        src_offset = lookup_df.loc[src, 'From Forest Hills (min)']
        dst_offset = lookup_df.loc[dst, 'From Forest Hills (min)']
    else:
        src_offset = lookup_df.loc[src, 'From Oak Grove (min)']
        dst_offset = lookup_df.loc[dst, 'From Oak Grove (min)']
    eta_src = (dep_time + timedelta(minutes=src_offset)).strftime("%H:%M")
    eta_dst = (dep_time + timedelta(minutes=dst_offset)).strftime("%H:%M")
    # Only show if ETA at source is after user_time
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