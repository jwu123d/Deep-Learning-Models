import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
df = pd.read_csv("sampleSubmission.csv")

# Generate random travel times from a normal distribution with the desired mean and standard deviation
mean_travel_time = 718
std_travel_time = 153
num_samples = len(df)
random_travel_times = np.random.normal(loc=mean_travel_time, scale=std_travel_time, size=num_samples)

# Update the "TRAVEL_TIME" column with the generated values
df["TRAVEL_TIME"] = random_travel_times.round().astype(int)

# Save the updated DataFrame to a new CSV file
df.to_csv("updated_dataset.csv", index=False)