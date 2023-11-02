import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("processed_trips.csv")
test_df = pd.read_csv('test_public.csv')
# Normalize 'TAXI_ID' and 'TIMESTAMP'
scaler = MinMaxScaler()
df[['TAXI_ID', 'TIMESTAMP']] = scaler.fit_transform(df[['TAXI_ID', 'TIMESTAMP']])

# Convert POLYLINE field into array of floats and pad sequences for consistent input size
max_len = 100
df['POLYLINE'] = df['POLYLINE'].apply(lambda x: np.array(eval(x)))
df['POLYLINE'] = df['POLYLINE'].apply(lambda x: x[:max_len] if len(x) > max_len else np.pad(x, (0, max_len - len(x)), 'constant', constant_values=0))

# Split data into features (X) and target (y)
X = df[['TAXI_ID', 'TIMESTAMP']].values
y = df['POLYLINE'].apply(lambda x: len(x)).values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TensorFlow Dataset for train and test data
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Define preprocessing function
def preprocess_data(x, y):
    x = tf.expand_dims(x, axis=0)  # Add a batch dimension
    return x, y

# Apply preprocessing function and batch the data
batch_size = 128
train_dataset = train_dataset.map(preprocess_data).batch(batch_size)
test_dataset = test_dataset.map(preprocess_data).batch(batch_size)

# Define LSTM model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(100, input_shape=(None, X_train.shape[1])))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mae', optimizer='adam')

# Fit model
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset, verbose=1)

# Save the model
model.save('my_model')

# Load the test dataset
test_public_df = pd.read_csv('test_public.csv')

# Normalize 'TAXI_ID' and 'TIMESTAMP' in the test dataset
test_public_df[['TAXI_ID', 'TIMESTAMP']] = scaler.transform(test_public_df[['TAXI_ID', 'TIMESTAMP']])

# Convert the normalized data to numpy array
test_public_features = test_public_df[['TAXI_ID', 'TIMESTAMP']].values

# Store a copy of the original 2D data
test_public_features_2d = test_public_features.copy()

# Reshape the data to 3D for the LSTM
test_public_features = test_public_features.reshape((test_public_features.shape[0], 1, test_public_features.shape[1]))

# Make predictions
test_public_predictions = model.predict(test_public_features).flatten()

# Create a DataFrame for the predicted lengths
pred_public_df = pd.DataFrame(test_public_predictions, columns=['LEN'])

# Add the 'TAXI_ID' column
# Use the 2D copy of the data when applying the inverse transformation
inverse_transform = scaler.inverse_transform(test_public_features_2d)[:, 0]
pred_public_df['TAXI_ID'] = inverse_transform

# Reorder the DataFrame to have 'TAXI_ID' as the first column
pred_public_df = pred_public_df[['TAXI_ID', 'LEN']]

# Save the predictions to a CSV file
pred_public_df.to_csv('my_public_predictions.csv', index=False)
LS
