import pandas as pd
import numpy as np
import random

# Set parameters to match your exact project constraint requirements 
np.random.seed(42)
n_samples = 500
n_features = 12

# Generate random mock baseline data 
data = np.random.randn(n_samples, n_features)

# Define target class rules based on feature boundaries
score = np.sum(data[:, :5], axis=1) + np.random.randn(n_samples) * 0.5
target = (score > 0).astype(int)

# Package column headers to fit the Streamlit script's reading layout
columns = [f"feature_{i+1}" for i in range(n_features)]
df = pd.DataFrame(data, columns=columns)
df['target'] = target
random_suffix = random.randint(1000, 9999)

# Save down directly into your project root folder
# df.to_csv(f"test_data_{random_suffix}.csv", index=False)
# print(f"Successfully generated test_data_{random_suffix}.csv with 12 features and 500 rows!")
df.to_csv(f"test_data.csv", index=False)
print(f"Successfully generated test_data.csv with 12 features and 500 rows!")
