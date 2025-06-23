import pandas as pd
import numpy as np
from scipy.stats import gmean

# Read the CSV file
df = pd.read_csv('filtered_orange_route.csv')

# Group by direction_id, day_type_id, and time_period_id
grouped = df.groupby(['direction_id', 'day_type_id', 'time_period_id'])

# Calculate arithmetic mean, geometric mean, and highest ons
results = grouped['total_ons'].agg([
    ('arithmetic_mean', 'mean'),
    ('geometric_mean', lambda x: gmean(x[x > 0]) if len(x[x > 0]) > 0 else np.nan),
    ('highest_ons', 'max')
]).reset_index()

# Display results
print(results)