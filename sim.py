import numpy as np
import matplotlib.pyplot as plt

# Simulate 10 2D paths
np.random.seed(0)  # for reproducibility
for i in range(10):
    latitudes = np.random.uniform(low=41.0, high=41.3, size=10)
    longitudes = np.random.uniform(low=-8.7, high=-8.5, size=10)
    plt.figure(figsize=(6, 4))
    plt.plot(longitudes, latitudes, marker='o')
    plt.title(f"Sample {i+1} with High Training Loss")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()


# Generate an array for your epochs
epochs = np.arange(1, 51)

# Parameters for the exponential decay
initial_loss = 0.5
decay_rate = 0.03

# Generate the exponential decay, but add some Gaussian noise
np.random.seed(0)
noise_std_dev = 0.02
noise = np.random.normal(scale=noise_std_dev, size=len(epochs))
loss = initial_loss * np.exp(-decay_rate * epochs) + noise

# Now plot the noisy exponential decay
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, label="RMSE")
plt.title("RMSE over Epochs")
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.grid(True)
plt.legend()
plt.show()
