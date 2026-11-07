# Problem Statement
Wisconsin Breast Cancer Dataset:
Build an automated machine learning model that accurately predicts whether a tumor is Benign (non-cancerous) or Malignant (cancerous).

# Dataset Description
1. The Input Features (X):
    The dataset is provided with highly detailed, multi-dimensional numerical features collected from the microscopic images of cell nuclei during a biopsy. These include attributes like:
    • Clump Thickness
    • Uniformity of Cell Size
    • Uniformity of Cell Shape
    • Marginal Adhesion
    • Single Epithelial Cell Size
    • Bare Nuclei
    • Bland Chromatin
    • Normal Nucleoli
    • Mitoses

2. The Target Outputs (y)

    The model must map those numeric features into one of two strict target classes:
    • 0 (or 2 in the raw data): Benign (safe mass, minimal intervention needed).
    • 1 (or 4 in the raw data): Malignant (cancerous mass, immediate clinical treatment required).

# Github Repository Link
https://github.com/pr701368/Classifiers

# Models used
Metrics obtained after running belwo models on Wisconsin Breast Cancer Dataset
|-------------------------------------------------------------------------------------------------------|
|Model	                                    |Accuracy	|Precision	|Recall	|F1-Score	|MCC	 | AUC  |
|-------------------------------------------------------------------------------------------------------|
|Logistic Regression	                    |0.8946	    |0.822	    |0.9151 |0.8661	    |0.7826	 |0.9219|
|Decision Tree Classifier	                |1	        |1	        |1	    |1	        |1	     |1     |
|K-Nearest Neighbor Classifier	            |0.9473	    |0.955	    |0.9009	|0.9272	    |0.8868	 |0.9917|
|Naive Bayes Classifier	                    |0.9947	    |0.986	    |1	    |0.993	    |0.9888	 |1     |
|Support Vector Machine Classifier	        |0.7504	    |0.6	    |0.9906	|0.7473	    |0.5946	 |0.7992|
|--------------------------------------------------------------------------------------------------------

# Observations

|----------------------------------------------------------------------------------------------------------|
|Model	                                    |Observations                                                  |
|----------------------------------------------------------------------------------------------------------|
|Logistic Regression	                    |Accuracy-0.8946,Precision-0.8220 and F1-Score-0.8661          |
|Decision Tree Classifier	                |Score 1 across all models. Example of Overfitting model       |
|K-Nearest Neighbor Classifier	            |Accuracy-0.9473, Precision-0.9860, Recall-0.9009              |
|Naive Bayes Classifier	                    |Accuracy-0.9947, Precision-0.9550, Recall -  1.0. Good model  |
|Support Vector Machine Classifier	        |Accuracy-0.7504, Precision-0.6, Recall-0.9906. Weak model     |
|----------------------------------------------------------------------------------------------------------|
Overall Winner of datset: Naive Bayes Classifier

# Streamlit app features:
a. Dataset upload option: By default when page loads, all or the selected models run on Wisconsin Breast Cancer Dataset. Also, a uplaod button available to upload datasets and apply predictions on it.
b. Model Select Option: A dropdown available to select models to run on dataset. By default all are selected. We can remove one or more models and apply the remaining models on the dataset.
c. Display of Evaluation Metrics: Shows metrics of models applied on dataset in th ewebpage
d. Confusion matrix or classification report: Contains a dropdown to select model. On selecting a model, a report displayed with cases predicted positive and negative

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
