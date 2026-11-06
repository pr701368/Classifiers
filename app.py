from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import subprocess
import sys

# Set up page configurations
st.set_page_config(page_title="ML Classifier Benchmark Dashboard", layout="wide")

st.title("📊 Machine Learning Classifier Benchmark Dashboard")
st.markdown("This dashboard tracks, evaluates, and directly compares the 5 required classification models across 6 evaluation metrics.")

# =====================================================================
# 1. INITIALIZE GLOBAL SESSION STATES
# =====================================================================
SAVE_DIR = "data_sample_files"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("model", exist_ok=True)

if "last_run_file" not in st.session_state:
    st.session_state.last_run_file = None
if "selected_models" not in st.session_state:
    st.session_state.selected_models = []

# Optional: Display historical information if it exists
if st.session_state.last_run_file:
    st.info(f"📁 Showing metrics computed for active uploaded dataset file: `{st.session_state.last_run_file}`")
    if st.button("🗑️ Clear Cached File Context"):
        st.session_state.last_run_file = None
        st.session_state.selected_models = []
        st.rerun()

st.divider()

# =====================================================================
# 2. BROWSE & UPLOAD TEST DATA UI PANEL
# =====================================================================
upload_file = st.file_uploader("Upload a Data (.csv) file", type=["csv"], key="file_uploader")

MODEL_OPTIONS = [
    "Logistic Regression", 
    "Decision Tree Classifier", 
    "K-Nearest Neighbor Classifier", 
    "Naive Bayes Classifier", 
    "Support Vector Machine Classifier"
]

# Track UI variables
selected_options = st.multiselect(
    label="Select Model to apply on the uploaded data (Optional):",
    options=MODEL_OPTIONS,
    default=MODEL_OPTIONS
)

# Handle Trigger Action Execution
if upload_file is not None:
    file_path = os.path.join(SAVE_DIR, upload_file.name)
    
    # Save the file out to local working directory safely
    with open(file_path, "wb") as f:
        f.write(upload_file.getbuffer())
    
    if st.button("🚀 Run selected models over above uploaded data"):
        st.info("Executing script processing pipelines...")
        
        # Save states globally inside session memory
        st.session_state.last_run_file = upload_file.name
        st.session_state.selected_models = selected_options
        
        # Parse data frames directly safely 
        try:
            test_df = pd.read_csv(file_path)
            X = test_df.iloc[:, :-1]
            y = test_df.iloc[:, -1]

            # Instantiation matching criteria maps
            models_pool = {
                "Logistic Regression": ("logistic_regression.joblib", LogisticRegression(max_iter=1000, random_state=42)),
                "Decision Tree Classifier": ("decision_tree_classifier.joblib", DecisionTreeClassifier(random_state=42)),
                "K-Nearest Neighbor Classifier": ("k-nearest_neighbor_classifier.joblib", KNeighborsClassifier(n_neighbors=5)),
                "Naive Bayes Classifier": ("naive_bayes_gaussian.joblib", GaussianNB()),
                "Support Vector Machine Classifier": ("support_vector_machine.joblib", SVC(probability=True, random_state=42))
            }

            # 4. Filter, train, and export ONLY chosen models matching user parameters
            for clean_name in selected_options:
                if clean_name in models_pool:
                    filename, model_obj = models_pool[clean_name]
                    model_obj.fit(X, y)
                    out_path = os.path.join("model", filename)
                    joblib.dump(model_obj, out_path)
            
            st.success("✅ Models trained and saved successfully!")
            st.rerun()  # Forces streamlit to refresh cleanly and run visual rendering below

        except Exception as e:
            st.error(f"❌ Processing Error: {str(e)}")

# =====================================================================
# 3. DYNAMIC EVALUATION ANALYSIS & MATRIX RENDER BLOCK
# =====================================================================
if st.session_state.last_run_file:
    active_filepath = os.path.join(SAVE_DIR, st.session_state.last_run_file)
    
    if os.path.exists(active_filepath):
        eval_df = pd.read_csv(active_filepath)
        X_test = eval_df.iloc[:, :-1]
        y_test = eval_df.iloc[:, -1]

        # Model mapping for saved artifacts evaluation
        model_files = {
            "Logistic Regression": "model/logistic_regression.joblib",
            "Decision Tree Classifier": "model/decision_tree_classifier.joblib",
            "K-Nearest Neighbor Classifier": "model/k-nearest_neighbor_classifier.joblib",
            "Naive Bayes Classifier": "model/naive_bayes_gaussian.joblib",
            "Support Vector Machine Classifier": "model/support_vector_machine.joblib"
        }

        results = []
        active_selections = st.session_state.selected_models if st.session_state.selected_models else MODEL_OPTIONS

        for clean_name in active_selections:
            file_path = model_files.get(clean_name)
            
            if file_path and os.path.exists(file_path):
                model = joblib.load(file_path)
                y_pred = model.predict(X_test)
                
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = y_pred

                # Custom metric formulas from scratch
                tp = sum((t == 1 and p == 1) for t, p in zip(y_test, y_pred))
                fp = sum((t == 0 and p == 1) for t, p in zip(y_test, y_pred))
                tn = sum((t == 0 and p == 0) for t, p in zip(y_test, y_pred))
                fn = sum((t == 1 and p == 0) for t, p in zip(y_test, y_pred))
                total = len(y_test)

                accuracy = (tp + tn) / total if total > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                mcc_num = (tp * tn) - (fp * fn)
                mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                mcc = mcc_num / mcc_den if mcc_den > 0 else 0.0

                pos_p = [p for t, p in zip(y_test, y_prob) if t == 1]
                neg_p = [p for t, p in zip(y_test, y_prob) if t == 0]
                
                auc = 0.0
                if len(pos_p) > 0 and len(neg_p) > 0:
                    satisfied_pairs = 0
                    for p_pos in pos_p:
                        for p_neg in neg_p:
                            if p_pos > p_neg:
                                satisfied_pairs += 1
                            elif p_pos == p_neg:
                                satisfied_pairs += 0.5
                    auc = satisfied_pairs / (len(pos_p) * len(neg_p))

                results.append({
                    "Model": clean_name,
                    "Accuracy": round(accuracy, 4),
                    "Precision": round(precision, 4),
                    "Recall (Sensitivity)": round(recall, 4),
                    "F1-Score": round(f1, 4),
                    "Matthews Corr Coeff (MCC)": round(mcc, 4),
                    "ROC AUC": round(auc, 4)
                })

        # Render matrix output cleanly out to standard presentation view grid
        if results:
            st.subheader("📊 Model Performance Benchmark Comparison Matrix")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df.set_index("Model"), use_container_width=True)
            st.markdown("**Quick Takeaway:** The model with the highest values across most metrics represents your strongest baseline classifier choice.")
            
            st.divider()
            
            # =====================================================================
            # 4. VISUAL METRICS CHARTS SECTION
            # =====================================================================
            st.subheader("📈 Graphical Performance Comparison")
            
            # Setup choice structure for plot options
            metric_columns = ["Accuracy", "Precision", "Recall (Sensitivity)", "F1-Score", "Matthews Corr Coeff (MCC)", "ROC AUC"]
            
            # Layout optimization split: Left for single filter view, Right for mass comparison
            chart_tab1, chart_tab2 = st.tabs(["Single Metric View", "Comprehensive Multi-Metric Comparison"])
            
            with chart_tab1:
                selected_metric = st.selectbox("Choose Metric to Plot:", options=metric_columns, index=3) # Default index 3 is F1-Score
                
                # Filter target values out into simple plot setup
                chart_df = results_df[["Model", selected_metric]].set_index("Model")
                st.bar_chart(chart_df, use_container_width=True)
                
            with chart_tab2:
                # Melt wide data format into native tall structures matching Streamlit chart patterns
                melted_df = pd.melt(results_df, id_vars=["Model"], value_vars=metric_columns, var_name="Metric", value_name="Value")
                
                # Display grouped comparisons natively
                st.bar_chart(
                    data=melted_df,
                    x="Model",
                    y="Value",
                    color="Metric",
                    stack=False,
                    use_container_width=True
)
else:
    st.error("⚠️ Data cache paths failed verification checks. Please try re-uploading file structures.")
