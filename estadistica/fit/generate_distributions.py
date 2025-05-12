import numpy as np
import csv
import scipy.stats as stats
import matplotlib.pyplot as plt

# Generate synthetic data: Normal
np.random.seed(0)
data = np.random.normal(loc=5, scale=2, size=1000)
data = data + np.random.uniform(low=-0.5, high=0.5, size=1000)
np.savetxt("data_0.csv", data)

# Generate synthetic data: Normal Truncated
np.random.seed(0)
data = np.random.normal(loc=5, scale=2, size=1000)
data = data + np.random.uniform(low=-0.5, high=0.5, size=1000)
data = np.maximum(data, 0)
np.savetxt("data_1.csv", data)

# Generate synthetic data: Normal Truncated
np.random.seed(0)
data = np.random.exponential(scale=10, size=1000)
data = data + np.random.uniform(low=0, high=1, size=1000)
np.savetxt("data_2.csv", data)