import pandas as pd
import random

# Get the number of lines in the file
num_lines = sum(1 for l in open('processed_trips.csv'))

# Calculate the number of lines to skip
skip = num_lines - 100000

# Get a list of indices to skip (excluding the first line as it is the header)
skip_idx = random.sample(range(1, num_lines), skip)

# Read the csv file with skipped rows
df = pd.read_csv('processed_trips.csv', skiprows=skip_idx)

df = df.sort_values(by='TAXI_ID')

df = df.drop(columns=['POLYLINE'])

df.to_csv('processed_trips.csv', index=False)