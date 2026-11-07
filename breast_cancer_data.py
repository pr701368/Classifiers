import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load the dataset directly from scikit-learn
data = load_breast_cancer(as_frame=True)
df = data.frame

# Optional: Map the target numbers (0, 1) to actual readable text
df['diagnosis_label'] = df['target'].map({0: 'malignant', 1: 'benign'})

# Save the dataset to your local directory
df.to_csv('breast_cancer_wisconsin.csv', index=False)

print("Dataset successfully saved as 'breast_cancer_wisconsin.csv'!")
