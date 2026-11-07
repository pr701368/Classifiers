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

from custom_models.logistic_regression import LogisticRegressionModel
from custom_models.naive_bayes import NaiveBayesModel
from custom_models.knn import KNNModel  
from custom_models.gini_decision_tree import GiniDecisionTreeModel
from custom_models.svm import SVMModel

# Set up page configurations
st.set_page_config(page_title="ML Classifier Benchmark Dashboard", layout="wide")

st.title("Machine Learning Classifier Benchmark Dashboard")
st.markdown("This dashboard tracks, evaluates, and directly compares the 5 required classification models across 6 evaluation metrics.")

# =====================================================================
# BULLETPROOF DATA CLEANING ENGINE
# =====================================================================
def clean_and_prepare_data(data_path):
    """
    Cleans datasets by dropping ID columns, mapping text targets ('M'/'B') to 0/1, 
    and converting all features safely to numeric values.
    """
    df = pd.read_csv(data_path)
    
    # 1. Drop common ID/Metadata columns by looking at names
    id_cols = [c for c in df.columns if c.lower() in ['id', 'patient_id', 'code', 'sample_code_number']]
    if id_cols:
        df = df.drop(columns=id_cols)
        
    # 2. Extract features and target separately
    X_df = df.iloc[:, :-1].copy()
    y_series = df.iloc[:, -1].copy()
    
    # 3. Handle Target Column (y): Map text strings ('M'/'B', 'malignant'/'benign') to binary digits
    y_series = y_series.astype(str).str.strip()
    unique_targets = y_series.unique()
    
    if len(unique_targets) == 2:
        # Sort so that 'B' or lower value maps to 0, 'M' or higher maps to 1
        unique_targets = sorted(unique_targets)
        mapping = {unique_targets[0]: 0, unique_targets[1]: 1}
        y = y_series.map(mapping).values
    else:
        # Fallback if already numeric values (like 2 and 4)
        y_numeric = pd.to_numeric(y_series, errors='coerce')
        unique_nums = np.unique(y_numeric[~np.isnan(y_numeric)])
        if len(unique_nums) == 2:
            y = np.where(y_numeric == unique_nums[0], 0, 1)
        else:
            y = y_numeric.fillna(0).astype(int).values

    # 4. Handle Features (X): Convert columns to numeric, replacing things like '?' with NaN
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
        
    # Fill any remaining missing text values or NaNs using column means
    X_df = X_df.ffill().bfill()
    X_df = X_df.fillna(0) # Ultimate fallback
    
    X = X_df.values.astype(float)
    
    return X, y

def run_model_training_pipeline(data_path, models_to_run):
    """Trains and saves selected models after processing clean arrays."""
    X, y = clean_and_prepare_data(data_path)

    # models_pool = {
    #         "Logistic Regression": ("logistic_regression.joblib", LogisticRegression(max_iter=1000, random_state=42)),
    #         "Decision Tree Classifier": ("decision_tree_classifier.joblib", DecisionTreeClassifier(random_state=42)),
    #         "K-Nearest Neighbor Classifier": ("k-nearest_neighbor_classifier.joblib", KNeighborsClassifier(n_neighbors=5)),
    #         "Naive Bayes Classifier": ("naive_bayes_gaussian.joblib", GaussianNB()),
    #         "Support Vector Machine Classifier": ("support_vector_machine.joblib", SVC(probability=True, random_state=42))
    #     }
    
    models_pool = {
        "Logistic Regression": ("logistic_regression.joblib", LogisticRegressionModel()),
        "Decision Tree Classifier": ("decision_tree_classifier.joblib", GiniDecisionTreeModel(max_depth=5)),
        "K-Nearest Neighbor Classifier": ("k-nearest_neighbor_classifier.joblib", KNNModel(k=5)),
        "Naive Bayes Classifier": ("naive_bayes_gaussian.joblib", NaiveBayesModel()),
        "Support Vector Machine Classifier": ("support_vector_machine.joblib", SVMModel(lr=0.001, lambda_param=0.01, n_iters=1000))
    }

    os.makedirs("model", exist_ok=True)
    for clean_name in models_to_run:
        if clean_name in models_pool:
            filename, model_obj = models_pool[clean_name]
            model_obj.fit(X, y)
            out_path = os.path.join("model", filename)
            joblib.dump(model_obj, out_path)

# =====================================================================
# 1. INITIALIZE GLOBAL SESSION STATES
# =====================================================================
SAVE_DIR = "data_sample_files"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("model", exist_ok=True)
DEFAULT_FILE = "breast_cancer_wisconsin.csv"

if "last_run_file" not in st.session_state:
    st.session_state.last_run_file = None
if "selected_models" not in st.session_state:
    st.session_state.selected_models = []

if st.session_state.last_run_file:
    if st.session_state.last_run_file == DEFAULT_FILE:
        st.info(f"💡 Dashboard auto-initialized with Default Dataset: `{DEFAULT_FILE}`")
    else:
        st.info(f"📁 Active custom file path context: `{st.session_state.last_run_file}`")
    
    if st.button("🗑️ Reset to Default View Context"):
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

selected_options = st.multiselect(
    label="Select Models to Evaluate:",
    options=MODEL_OPTIONS,
    default=MODEL_OPTIONS
)

if upload_file is not None:
    file_path = os.path.join(SAVE_DIR, upload_file.name)
    with open(file_path, "wb") as f:
        f.write(upload_file.getbuffer())
    
    if st.button("🚀 Run selected models over above uploaded data"):
        st.info("Executing robust processing pipelines...")
        try:
            run_model_training_pipeline(file_path, selected_options)
            st.session_state.last_run_file = upload_file.name
            st.session_state.selected_models = selected_options
            st.success("✅ Models successfully trained!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Processing Error: {str(e)}")
else:
    if st.session_state.last_run_file is None:
        if os.path.exists(DEFAULT_FILE):
            run_model_training_pipeline(DEFAULT_FILE, MODEL_OPTIONS)
            st.session_state.last_run_file = DEFAULT_FILE
            st.session_state.selected_models = MODEL_OPTIONS
            # st.rerun()
        else:
            st.warning(f"⚠️ `{DEFAULT_FILE}` not found. Please upload a CSV manually.")

# =====================================================================
# 3. DYNAMIC EVALUATION ANALYSIS & MATRIX RENDER BLOCK
# =====================================================================
if st.session_state.last_run_file:
    active_filepath = DEFAULT_FILE if st.session_state.last_run_file == DEFAULT_FILE else os.path.join(SAVE_DIR, st.session_state.last_run_file)
    
    if os.path.exists(active_filepath):
        # Safely clean evaluation data through the identical engineering logic
        X_test, y_test = clean_and_prepare_data(active_filepath)

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

                # Custom metric equations from scratch
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

        if results:
            st.subheader(DEFAULT_FILE if st.session_state.last_run_file == DEFAULT_FILE else st.session_state.last_run_file)
            st.subheader("📊 Model Performance Benchmark Comparison Matrix")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df.set_index("Model"), use_container_width=True)
            st.markdown("Quick Takeaway: The model with the highest values across most metrics represents your strongest baseline classifier choice.")
            st.divider()
            # CONFUSION MATRIX & CLASSIFICATION REPORT PANEL
            st.subheader("📋 Detailed Diagnostic Reports & Confusion Matrices")
            st.markdown("Select a specific model below to view its complete classification breakdown matrix.")
            
            # Dropdown menu to select model dynamically
            report_model = st.selectbox("Choose Model to Inspect Details:", options=active_selections)
            
            # Re-run logic for the single selected model to populate raw quadrants
            file_path = model_files.get(report_model)
            if file_path and os.path.exists(file_path):
                selected_model_obj = joblib.load(file_path)
                y_pred_sel = selected_model_obj.predict(X_test)
                
                # Compute raw diagnostic quadrants
                tp = sum((t == 1 and p == 1) for t, p in zip(y_test, y_pred_sel))
                fp = sum((t == 0 and p == 1) for t, p in zip(y_test, y_pred_sel))
                tn = sum((t == 0 and p == 0) for t, p in zip(y_test, y_pred_sel))
                fn = sum((t == 1 and p == 0) for t, p in zip(y_test, y_pred_sel))
                
                # Dynamic performance parameters extraction
                prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                f1_0 = 2 * (prec_0 * rec_0) / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0
                
                prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_1 = 2 * (prec_1 * rec_1) / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0.0

                # Layout Split: Left for Matrix Visual Grid, Right for Classification Report
                report_col1, report_col2 = st.columns(2)
                
                with report_col1:
                    st.write("**Visual Confusion Matrix Layout Grid**")
                    
                    # Create a neat grid layout structure using Pandas
                    matrix_data = {
                        "Predicted Benign (0)": [tn, fn],
                        "Predicted Malignant (1)": [fp, tp]
                    }
                    matrix_df = pd.DataFrame(matrix_data, index=["Actual Benign (0)", "Actual Malignant (1)"])
                    
                    # Highlight cell blocks with custom colors based on size
                    st.dataframe(matrix_df.style.background_gradient(cmap="Blues"), use_container_width=True)
                    
                    st.caption(f"Total Samples Analyzed: {len(y_test)} | Correct: {tn+tp} | Incorrect: {fp+fn}")
                    
                with report_col2:
                    st.write("**Text Classification Report Outline**")
                    
                    # Construct structural assignment text table manually string by string
                    report_text = (
                        f"{"Class Label":<20} {"Precision":<12} {"Recall":<12} {"F1-Score":<12} {"Support":<12}\n"
                        f"{'-'*70}\n"
                        f"{"Benign (0)":<20} {prec_0:<12.4f} {rec_0:<12.4f} {f1_0:<12.4f} {tn+fp:<12}\n"
                        f"{"Malignant (1)":<20} {prec_1:<12.4f} {rec_1:<12.4f} {f1_1:<12.4f} {tp+fn:<12}\n"
                        f"{'-'*70}\n"
                        f"{"Macro Average":<20} {((prec_0+prec_1)/2):<12.4f} {((rec_0+rec_1)/2):<12.4f} {((f1_0+f1_1)/2):<12.4f} {len(y_test):<12}"
                    )
                    st.code(report_text, language="text")

            st.divider()
            st.subheader("📈 Graphical Performance Comparison")
            metric_columns = ["Accuracy", "Precision", "Recall (Sensitivity)", "F1-Score", "Matthews Corr Coeff (MCC)", "ROC AUC"]
            chart_tab1, chart_tab2 = st.tabs(["Single Metric View", "Comprehensive Multi-Metric Comparison"])
            with chart_tab1:
                selected_metric = st.selectbox("Choose Metric to Plot:", options=metric_columns, index=3)
                chart_df = results_df[["Model", selected_metric]].set_index("Model")
                st.bar_chart(chart_df, use_container_width=True)
            with chart_tab2:
                melted_df = pd.melt(results_df, id_vars=["Model"], value_vars=metric_columns, var_name="Metric", value_name="Value")
                st.bar_chart(
                data=melted_df,
                x="Model",
                y="Value",
                color="Metric",
                stack=False,
                use_container_width=True
                )