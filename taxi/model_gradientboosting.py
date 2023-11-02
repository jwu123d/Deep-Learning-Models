import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# load the dataset
df = pd.read_csv('train.csv')

# Convert timestamp to meaningful features
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], unit='s')
df['WEEKDAY'] = df['TIMESTAMP'].dt.weekday
df['HOUR'] = df['TIMESTAMP'].dt.hour

# Calculate travel time
df['TRAVEL_TIME'] = df['POLYLINE'].apply(lambda x: (x.count(',')//2)*15) 

# Drop columns that are not necessary
df = df.drop(['TRIP_ID', 'TIMESTAMP', 'POLYLINE'], axis=1)

# Fill NA values
df['ORIGIN_CALL'] = df['ORIGIN_CALL'].fillna(0)
df['ORIGIN_STAND'] = df['ORIGIN_STAND'].fillna(0)

# Convert boolean to int
df['MISSING_DATA'] = df['MISSING_DATA'].astype(int)

# Label encode categorical features
le_call_type = LabelEncoder()
df['CALL_TYPE'] = le_call_type.fit_transform(df['CALL_TYPE'])

le_day_type = LabelEncoder()
df['DAY_TYPE'] = le_day_type.fit_transform(df['DAY_TYPE'])

# Split the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(df.drop(['TRAVEL_TIME'], axis=1), df['TRAVEL_TIME'], test_size=0.2, random_state=42)
model = GradientBoostingRegressor(verbose=1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: %f" % (rmse))
