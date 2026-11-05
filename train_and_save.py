import os
import pandas as pd
import joblib

# 1. Import models from scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 2. Check if test data exists, if not create a quick mock version
if not os.path.exists("test_data.csv"):
    import numpy as np
    np.random.seed(42)
    mock_X = np.random.randn(500, 12)
    mock_y = (np.sum(mock_X[:, :5], axis=1) > 0).astype(int)
    cols = [f"feature_{i+1}" for i in range(12)]
    df = pd.DataFrame(mock_X, columns=cols)
    df['target'] = mock_y
    df.to_csv("test_data.csv", index=False)
    print("✅ Created missing test_data.csv file.")

# 3. Load dataset
df = pd.read_csv("test_data.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 4. Initialize the 5 models with exact settings matching your app.py
models = {
    "logistic_regression.joblib": LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree_classifier.joblib": DecisionTreeClassifier(random_state=42),
    "k-nearest_neighbor_classifier.joblib": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes_gaussian.joblib": GaussianNB(),
    "support_vector_machine.joblib": SVC(probability=True, random_state=42)
}

# 5. Create the model directory if missing
os.makedirs("model", exist_ok=True)

# 6. Train and export each model file
print("\nTraining and saving models...")
for filename, model in models.items():
    model.fit(X, y)
    file_path = os.path.join("model", filename)
    joblib.dump(model, file_path)
    print(f" Saved: {file_path}")

print("\n🎉 Success! Refresh your Streamlit browser tab to see your dashboard work.")
