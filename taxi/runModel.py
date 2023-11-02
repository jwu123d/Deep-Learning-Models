import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

train_df = pd.read_csv('processed_trips.csv')
test_df = pd.read_csv('test_public.csv')


# Select features and target
features = ['TAXI_ID', 'TIMESTAMP','ORIGIN_STAND']
target = ['LEN']

X = train_df[features]
y = train_df[target]

# Split the dataset into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Instantiate a Random Forest regressor
rf = RandomForestRegressor(n_estimators=200, random_state=42)

# Train the model
rf.fit(X_train_scaled, y_train.values.ravel())  # ravel() is used to convert the column vector y to a 1D array

# Validate the model
val_predictions = rf.predict(X_val_scaled)

# Print validation score
print('Validation Score:', rf.score(X_val_scaled, y_val))

# Normalize the test features
test_features = scaler.transform(test_df[features])

# Make predictions on the test set
test_predictions = rf.predict(test_features)

# Create a DataFrame for the predicted lengths
pred_df = pd.DataFrame(test_predictions, columns=['TRAVEL_TIME'])

# Add the 'TAXI_ID' column
pred_df['TRIP_ID'] = test_df['TRIP_ID']

# Reorder the DataFrame to have 'TAXI_ID' as the first column
pred_df = pred_df[['TRIP_ID', 'TRAVEL_TIME']]

# Save the predictions to a CSV file
pred_df.to_csv('my_rf_predictions.csv', index=False)
