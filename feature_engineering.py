import pandas as pd
import numpy as np

# Assuming the csv file is named 'trips.csv' 
df = pd.read_csv('train.csv')

df = df[~df['DAY_TYPE'].isin(['B', 'C'])]

# Drop columns 'TRIP_ID', 'ORIGIN_CALL', 'ORIGIN_STAND'
df = df.drop(columns=['DAY_TYPE','TRIP_ID', 'ORIGIN_CALL'])

# Drop rows with missing data
df = df[df['MISSING_DATA'] == False]
df = df.drop(columns=['MISSING_DATA'])


def polyline_to_trip_duration(polyline):
      return max(polyline.count("[") - 2, 0) * 15

# This code creates a new column, "LEN", in our dataframe. The value is
# the (polyline_length - 1) * 15, where polyline_length = count("[") - 1
df["LEN"] = df["POLYLINE"].apply(polyline_to_trip_duration)


# Remove outliers
mean = df['LEN'].mean()
std = df['LEN'].std()
df = df[np.abs(df['LEN'] - mean) <= (3 * std)]


df.to_csv('processed_trips.csv', index=False)