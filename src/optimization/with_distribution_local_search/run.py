import streamlit as st
import pandas as pd
import numpy as np
import json

# ---------------- Load JSON ----------------
@st.cache_data
def load_json(path="data.json"):
    with open(path, "r") as f:
        return pd.DataFrame(json.load(f))

df = load_json()

# ---------------- Preprocessing ----------------
df = df[(df["day_type_id"] == "day_type_01") & (df["direction_id"] == 0)]
df["time_period_id"] = df["time_period_id"].astype(str)

agg_df = df.groupby("time_period_id").agg({
    "geometric_mean": "mean",
    "highest_ons": "mean"
}).reset_index()

# Time periods and durations
time_periods = sorted(agg_df["time_period_id"].tolist())
durations = [2, 3, 6.5, 3, 3.5, 3, 2, 2, 2, 2, 2]  # Extend if more periods
duration_map = dict(zip(time_periods, durations[:len(time_periods)]))

# ---------------- Optimization Setup ----------------
geo_load = agg_df.set_index("time_period_id").reindex(time_periods)["geometric_mean"].values
max_ons = agg_df.set_index("time_period_id").reindex(time_periods)["highest_ons"].values

train_capacity = 50
max_total_trips = 100
initial_freqs = [4] * len(time_periods)
min_freq = 1
max_freq = 15

# ---------------- Cost Function ----------------
def cost(frequencies):
    wait_cost = np.sum(60 / np.array(frequencies))
    crowding_cost = 0
    overprovision_cost = 0

    for i, f in enumerate(frequencies):
        period = time_periods[i]
        duration = duration_map.get(period, 3)
        cap = f * train_capacity * duration
        demand = max_ons[i] #geo_load[i] + 0.2 * max_ons[i]

        if demand > cap:
            crowding_cost += demand - cap
        elif cap > 1.5 * demand:
            overprovision_cost += cap - 1.5 * demand

    total_trips = sum([frequencies[i] * duration_map.get(time_periods[i], 3) for i in range(len(frequencies))])
    trip_penalty = (total_trips - max_total_trips) ** 2 if total_trips > max_total_trips else 0

    return wait_cost + 2 * crowding_cost + 0.5 * overprovision_cost + 2.0 * trip_penalty

# ---------------- Hill Climbing ----------------
def hill_climb(frequencies, iterations=1000):
    current = frequencies.copy()
    best_cost = cost(current)
    for _ in range(iterations):
        i = np.random.randint(0, len(frequencies))
        direction = np.random.choice([-1, 1])
        neighbor = current.copy()
        neighbor[i] = np.clip(neighbor[i] + direction, min_freq, max_freq)
        new_cost = cost(neighbor)
        if new_cost < best_cost:
            current = neighbor
            best_cost = new_cost
    return current

# ---------------- Run Optimization ----------------
optimized_freqs = hill_climb(initial_freqs)

# ---------------- Build Result Table ----------------
result_df = pd.DataFrame({
    "Time Period": time_periods,
    "Duration (hrs)": [duration_map[p] for p in time_periods],
    "Geometric Mean Load": geo_load,
    "Highest ONS": max_ons,
    "Optimized Trains/Hour": optimized_freqs,
    "Total Trips": np.array(optimized_freqs) * [duration_map[p] for p in time_periods],
    "Total Capacity": np.array(optimized_freqs) * [duration_map[p] for p in time_periods] * train_capacity
})

# ---------------- Streamlit UI ----------------
st.title("Orange Line Frequency Optimizer")

selected_time = st.selectbox("Select Time Period", time_periods)

row = result_df[result_df["Time Period"] == selected_time].iloc[0]
st.markdown(f"###`{selected_time}`")
st.markdown(f"**Optimized Frequency:** 🚆 {int(row['Optimized Trains/Hour'])} trains/hour")
st.markdown(f"**Geometric Mean Load:** {int(row['Geometric Mean Load'])}")
st.markdown(f"**Highest ONS:** {int(row['Highest ONS'])}")
st.markdown(f"**Estimated Capacity:** {int(row['Total Capacity'])}")

if st.checkbox("Show Full Optimized Schedule"):
    st.dataframe(result_df)
