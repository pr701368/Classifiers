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
# 1. BROWSE & UPLOAD TEST DATA
# =====================================================================

SAVE_DIR = "data_sample_files"
os.makedirs(SAVE_DIR, exist_ok=True)
# 1. Check if we already have a successfully run file in memory
if "last_run_file" not in st.session_state:
    st.session_state.last_run_file = None
if "last_run_output" not in st.session_state:
    st.session_state.last_run_output = None
if "script_executed" not in st.session_state:
    st.session_state.script_executed = False

# Display historical data if it exists
if st.session_state.last_run_file:
    st.info(f"📁 Showing data for last executed file: `{st.session_state.last_run_file}`")
    st.subheader("Last Execution Terminal Output:")
    st.code(st.session_state.last_run_output)
    
    # Optional: Clear memory button
    if st.button("Clear Saved Data"):
        st.session_state.last_run_file = None
        st.session_state.last_run_output = None
        st.rerun()

st.divider()
upload_file = st.file_uploader("Upload a Data (.csv) file", type=["csv"], key="file_uploader")

if upload_file is not None:
    # Update state upon success
    st.session_state.script_executed = False
    file_path = os.path.join(SAVE_DIR, upload_file.name)
    with open(file_path, "wb") as f:
        f.write(upload_file.getbuffer())
    
    st.success(f"Saved: `{file_path}`")
    if not st.session_state.script_executed:
        if st.button("🚀 Run Logistic,Decision Tree,K-Neighbour,Naive Bayes,SVM models over above uploaded data"):
            st.info("Executing script...")
            try:
                script_to_run = "train_and_save.py"
                result = subprocess.run(
                    [sys.executable, script_to_run], 
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                # Update state upon success
                st.session_state.script_executed = True
                st.success("✅ Script finished successfully! Reloading page...")
                st.session_state.last_run_file = upload_file.name
                st.session_state.last_run_output = result.stdout if result.stdout else "Script executed with no output."
                
                # Clear file uploader state and refresh logic
                st.rerun()
                    
                # Inject JavaScript to reload the browser window completely
                components.html(
                    """
                    <script>
                        window.parent.location.reload();
                    </script>
                    """,
                    height=0,
                )
                    
            except subprocess.CalledProcessError as e:
                st.error("❌ Script failed with an error!")
                st.code(e.stderr)
# =====================================================================
# 1. LOAD TEST DATA & ARTIFACTS
# =====================================================================
@st.cache_data
def load_test_data():
    if st.session_state.last_run_file == None:
    # Looks for test_data.csv in your root folder
        script_to_run_test_data = "testdata.py"
        result = subprocess.run(
                    [sys.executable, script_to_run_test_data], 
                            capture_output=True, 
                            text=True, 
                            check=True
                        )
        if os.path.exists("test_data.csv"):
            df = pd.read_csv("test_data.csv")
            return df
        else:
            st.error("⚠️ `test_data.csv` not found in the project root directory! Please run your training pipeline first.")
            return None
    else:
        if os.path.exists(os.path.join(SAVE_DIR, st.session_state.last_run_file)): #st.session_state.last_run_file
            df = pd.read_csv(os.path.join(SAVE_DIR, st.session_state.last_run_file))
            return df
        else:
            st.error(f"⚠️ {os.path.join(SAVE_DIR, st.session_state.last_run_file)} not found in the project root directory! Please run your trining pipeline first.")
            return None


test_df = load_test_data() if st.session_state.last_run_file is None else pd.read_csv(os.path.join(SAVE_DIR, st.session_state.last_run_file))


if test_df is not None:
    # Separate features and target (Assuming the last column is the target/label)
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    # Map filename anchors to clean dashboard display titles
    model_files = {
        "Logistic Regression": "model/logistic_regression.joblib",
        "Decision Tree Classifier": "model/decision_tree_classifier.joblib",
        "K-Nearest Neighbor Classifier": "model/k-nearest_neighbor_classifier.joblib",
        "Naive Bayes (Gaussian)": "model/naive_bayes_gaussian.joblib",
        "Support Vector Machine": "model/support_vector_machine.joblib"
    }

    # =====================================================================
    # 2. RUN COMPUTATIONS & INFERENCE FROM SAVED ARTIFACTS
    # =====================================================================
    # Manual evaluation metric formulas from scratch to mirror your assignment logic
    results = []
    
    # Verify that the model directory exists
    missing_models = [name for name, path in model_files.items() if not os.path.exists(path)]
    
    if missing_models:
        st.warning(f"⚠️ Missing trained model files: {', '.join(missing_models)}. Please make sure your model files are saved in the `model/` folder.")
    else:
        for model_name, file_path in model_files.items():
            # Load the saved model file directly from disk
            model = joblib.load(file_path)
            
            # Predict values
            y_pred = model.predict(X_test)
            
            # Use safety conditions to check if model supports continuous probability output
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred

            # Matrix quadrant parsing
            tp = sum((t == 1 and p == 1) for t, p in zip(y_test, y_pred))
            fp = sum((t == 0 and p == 1) for t, p in zip(y_test, y_pred))
            tn = sum((t == 0 and p == 0) for t, p in zip(y_test, y_pred))
            fn = sum((t == 1 and p == 0) for t, p in zip(y_test, y_pred))
            total = len(y_test)

            # Mathematical metrics processing
            accuracy = (tp + tn) / total if total > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            mcc_num = (tp * tn) - (fp * fn)
            mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = mcc_num / mcc_den if mcc_den > 0 else 0.0

            # Direct Rank-Sum execution for non-library AUC computation
            pos_p = [p for t, p in zip(y_test, y_prob) if t == 1]
            neg_p = [p for t, p in zip(y_test, y_prob) if t == 0]
            if pos_p and neg_p:
                greater = sum((p > n) for p in pos_p for n in neg_p) + 0.5 * sum((p == n) for p in pos_p for n in neg_p)
                auc = greater / (len(pos_p) * len(neg_p))
            else:
                auc = 0.5

            results.append({
                "Model Framework": model_name,
                "Accuracy": accuracy,
                "AUC Score": auc,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "MCC Score": mcc
            })

        # Convert evaluation records into structured DataFrame formats
        summary_df = pd.DataFrame(results)

        # =====================================================================
        # 3. INTERACTIVE LAYOUT RENDERING
        # =====================================================================
        st.subheader("📋 Complete Performance Metrics Comparison Matrix")
        st.dataframe(summary_df.style.highlight_max(axis=0, color="#d4edda", subset=summary_df.columns[1:]), use_container_width=True)

        st.markdown("---")
        st.subheader("📈 Visual Comparison Charts")
        
        # Dropdown selection allows graders to instantly chart different performance variables
        metric_choice = st.selectbox(
            "Select Evaluation Metric to visualize:",
            ["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC Score"]
        )

        # Generate responsive structural chart bars natively with Streamlit
        chart_data = summary_df.set_index("Model Framework")[[metric_choice]]
        st.bar_chart(chart_data)

        # Quick summary breakdown metrics box logic
        st.markdown("### 🏆 Top Performing System Overview")
        best_model = summary_df.loc[summary_df[metric_choice].idxmax()]["Model Framework"]
        best_value = summary_df[metric_choice].max()
        st.success(f"**{best_model}** holds the highest metric baseline score for **{metric_choice}** with a rating of **{best_value:.4f}**")
