# MBTA OPTIMAL SCHEDULE FINDER ​

This project implements an AI-driven train scheduling optimization system using passenger flow data and GTFS (General Transit Feed Specification) data. The system aims to optimize train frequencies based on observed passenger patterns during different time periods.

## Project Overview

The system uses local search techniques (Hill Climbing/Simulated Annealing/Genetic Algorithm) to optimize train scheduling by:
- Analyzing passenger flow data (ons/offs) across different time periods
- Processing GTFS data to determine available train slots
- Optimizing train allocation to minimize passenger wait times and overload
- Make a webapp with the most optimal schedule

https://github.com/user-attachments/assets/57b937bd-7f78-4600-b0fd-52fb819783f2


  

## Project Structure

```
├── data/
│   ├── passenger_flow/         # Passenger flow data
│   └── gtfs/                   # GTFS data files
├── src/
│   ├── data_processing/        # Data preprocessing modules
│   ├── optimization/           # AI optimization algorithms
│   ├── visualization/          # Data visualization tools
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Data Requirements

### Passenger Flow Data
- Format: CSV
- Required columns:
  - stop_name
  - parent_station
  - day_type (weekday/weekend)
  - time_period (AM_RUSH, PM_RUSH, etc.)
  - total_ons
  - total_offs

### GTFS Data
Required files:
- stop_times.txt
- trips.txt
- calendar.txt

## Implementation Steps

### 1. Data Preprocessing
- Load and clean passenger flow data
- Process GTFS data to extract train schedules
- Create unified data structure:
  ```
  stop | day_type | time_period | total_ons | total_offs
  ```

### 2. Train Slot Analysis
- Calculate availabletime slots
- Determine maximum train capacity per period
- Create slot allocation structure:
  ```
  day_type | time_period | max_slots
  ```

### 3. Optimization Implementation
- Define cost function:
  ```
  cost = (predicted_load_per_train - train_capacity)²
  ```
- Implement local search algorithms:
  - Hill Climbing
  - Simulated Annealing
  - Genetic Algorithm
- Handle constraints:
  - Maximum trains per time slot
  - Peak hour requirements

### 4. Time Period Classification
- Define peak periods:
  - AM_RUSH
  - PM_RUSH
- Define off-peak periods:
  - MIDDAY
  - EVENING

### 5. Output Generation
Generate optimization results including:
- Time period analysis
- Train allocation recommendations
- Load distribution statistics
- Visualization of results

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- gtfs-kit

## Setup Instructions

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place data files in appropriate directories
5. Run preprocessing scripts
6. Execute optimization algorithms

## Usage

run the whole pipeline using:

python main.py  # will run the optimization in depth

run the most optimal solution based webapp:
streamlit run src/visualization/app.py


## Expected Output

The system will generate:
1. Optimized train schedules
2. Load distribution analysis
3. Visualization of:
   - Demand vs. capacity
   - Optimized train counts
   - Peak vs. off-peak patterns
   - Cost comparison between all three algorithms

## Performance Metrics

- Passenger wait time reduction
- Train capacity utilization
- Overload minimization
- Schedule efficiency
- Overall cost

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
