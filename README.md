# Machine Learning Classifier Benchmark Application

An interactive benchmarking suite that loads **5 saved classifier models** from disk and evaluates them across **6 core classification metrics** using a live Streamlit dashboard interface.

---

## 📂 Project Structure

```text
project-folder/
├── app.py                  # Streamlit dashboard interface code
├── requirements.txt         # Core environment library requirements
├── README.md               # Setup and project guide documentation
├── test_data.csv           # Evaluation hold-out dataset matrix
└── model/                  # Directory containing serialized model pipelines
    ├── logistic_regression.joblib
    ├── decision_tree_classifier.joblib
    ├── k-nearest_neighbor_classifier.joblib
    ├── naive_bayes_gaussian.joblib
    └── support_vector_machine.joblib
```

---

## ⚙️ Setup and Installation

Follow these steps to run the pipeline on your local system environment:

### Step 1: Clone or Navigate to the Folder
Open your terminal application and route into your active project root directory folder:
```bash
cd path/to/your/project-folder
```

### Step 2: Install Libraries
Install the designated runtime dependencies listed in the requirements file using `pip`:
```bash
pip install -r requirements.txt
```

---

## 🚀 Execution Guide

### Launching the Dashboard App
Start the interactive local hosting server by executing the dashboard command:
```bash
streamlit run app.py
```

Once running, the terminal will provide a local web URL (typically `http://localhost:8501`). Open this address inside your browser to access the complete metric comparison grid and charts.

---

## 📊 Implemented System Features

### Evaluated Model Paradigms
1. **Logistic Regression** (Linear parametric probability categorization)
2. **Decision Tree Classifier** (Axis-aligned Gini structural branches)
3. **K-Nearest Neighbor Classifier** (Instance distance proximity mapping)
4. **Gaussian Naive Bayes** (Independent feature Bayes probability curves)
5. **Support Vector Machine** (Maximum-margin structural separating hyperplanes)

### Evaluated Mathematical Performance Parameters
* **Accuracy**
* **AUC Score** (Computed via non-library Wilcoxon rank-sum ranking evaluation logic)
* **Precision**
* **Recall**
* **F1-Score** (Harmonic evaluation mean parameter metric)
* **Matthews Correlation Coefficient (MCC Score)**
