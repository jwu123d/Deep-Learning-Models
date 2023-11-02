import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# load data
df = pd.read_csv("processed_trips.csv")

print("File Reading complete")

# assuming your target column is 'LEN'
X = df.drop('LEN', axis=1).values
y = df['LEN'].values.astype('float32')  
print("X and Y processing")

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train test splited")

# reshape input to be 3D [samples, timesteps, features]



def sum_coordinates(coord_string):
    try:    # Use ast.literal_eval to safely parse the string as a Python literal
        coord_list = ast.literal_eval(coord_string)
    
        # Sum all coordinates
        return np.sum(coord_list)
    except (SyntaxError, ValueError):
        # Handle cases where the input cannot be evaluated
        return np.nan
    
X_train_df = pd.DataFrame(X_train)
X_train_df = X_train_df.applymap(sum_coordinates)
X_train = X_train_df.values.astype('float32')
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
X_test = X_test.astype('float32')
y_test = y_test.astype('float32')


X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print("Reshape completed")
# define LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit model
print("Start training")
history = model.fit(X_train, y_train, epochs=50, batch_size=72, validation_data=(X_test, y_test), verbose=1, shuffle=False)
# save model
model.save('trained_model.h5')