
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Bank Marketing ML Dashboard",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Marketing Classification Dashboard")

st.markdown("""
### Machine Learning Model Evaluation

An interactive application to evaluate five
classification algorithms on Bank Marketing data.

**Dataset:** UCI Bank Marketing Dataset
**Problem:** Predict customer term deposit subscription.
""")

st.divider()

# ==================================================
# 2. MODEL CONFIGURATION
# ==================================================

model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "K-Nearest Neighbors": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}


@st.cache_resource
def load_models():

    loaded_models = {}

    for name, path in model_files.items():

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing model file: {path}"
            )

        loaded_models[name] = joblib.load(path)

    return loaded_models


try:
    models = load_models()

    st.success(
        "All five trained ML models loaded successfully!"
    )

except Exception as e:

    st.error(f"Model loading failed: {e}")
    st.stop()


# ==================================================
# 3. DATASET UPLOAD
# ==================================================

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test_data.csv",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        test_df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Preview")

        st.dataframe(test_df.head())

        st.write("Dataset dimensions:", test_df.shape)

        # ------------------------------------------
        # CHECK REQUIRED FEATURES
        # ------------------------------------------

        required_columns = (
            models["Logistic Regression"]
            .named_steps["preprocessing"]
            .feature_names_in_
            .tolist()
        )

        expected_columns = required_columns + ["y"]

        if "y" not in test_df.columns:

            st.error(
                "Target column 'y' is missing."
            )

            st.stop()

        missing_columns = set(expected_columns) - set(test_df.columns)

        extra_columns = set(test_df.columns) - set(expected_columns)

        if missing_columns or extra_columns:

            st.error(
                f"Missing columns: {missing_columns}. "
                f"Unexpected columns: {extra_columns}."
            )

            st.stop()

        # ------------------------------------------
        # HANDLE MISSING VALUES
        # ------------------------------------------

        if test_df[required_columns].isnull().values.any():

            st.error(
                "Input features contain missing values. "
                "Please check your test CSV."
            )

            st.stop()

        # ------------------------------------------
        # TARGET COLUMN CONVERSION
        # ------------------------------------------

        if pd.api.types.is_numeric_dtype(test_df["y"]):

            y_test = pd.to_numeric(
                test_df["y"],
                errors="coerce"
            )

        else:

            y_test = (
                test_df["y"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({
                    "no": 0,
                    "yes": 1
                })
            )

        if y_test.isnull().any():

            st.error(
                "Target column must contain only "
                "0/1 or yes/no values."
            )

            st.stop()

        if not set(y_test.unique()).issubset({0, 1}):

            st.error(
                "Invalid target values. "
                "Expected 0 or 1."
            )

            st.stop()

        if y_test.nunique() != 2:

            st.error(
                "Both target classes must be present "
                "for AUC calculation."
            )

            st.stop()

        # ------------------------------------------
        # PREPARE INPUT FEATURES
        # ------------------------------------------

        X_test = test_df[required_columns]

        y_test = y_test.astype(int)

        st.success(
            "Input features and target column validated!"
        )

        # ==========================================
        # 4. MODEL SELECTION
        # ==========================================

        st.divider()

        st.header("2. Select Classification Model")

        selected_model = st.selectbox(
            "Choose an ML algorithm",
            list(models.keys())
        )

        model = models[selected_model]

        # ==========================================
        # 5. MODEL PREDICTIONS
        # ==========================================

        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(X_test)[:, 1]

        # ==========================================
        # 6. EVALUATION METRICS
        # ==========================================

        st.divider()

        st.header("3. Model Evaluation Results")

        accuracy = accuracy_score(
            y_test, y_pred
        )

        auc = roc_auc_score(
            y_test, y_prob
        )

        precision = precision_score(
            y_test, y_pred, zero_division=0
        )

        recall = recall_score(
            y_test, y_pred, zero_division=0
        )

        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )

        mcc = matthews_corrcoef(
            y_test, y_pred
        )

        # Display metrics

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        col2.metric(
            "AUC Score",
            f"{auc:.4f}"
        )

        col3.metric(
            "Precision",
            f"{precision:.4f}"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Recall",
            f"{recall:.4f}"
        )

        col5.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

        col6.metric(
            "MCC Score",
            f"{mcc:.4f}"
        )

        # ==========================================
        # 7. CONFUSION MATRIX
        # ==========================================

        st.divider()

        st.header("4. Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            ax=ax
        )

        ax.set_title(
            f"Confusion Matrix - {selected_model}"
        )

        ax.set_xlabel("Predicted Label")

        ax.set_ylabel("Actual Label")

        st.pyplot(fig)

        plt.close(fig)

        # ==========================================
        # 8. CLASSIFICATION REPORT
        # ==========================================

        st.divider()

        st.header("5. Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["No", "Yes"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df)

        # ==========================================
        # 9. PREDICTION RESULTS
        # ==========================================

        st.divider()

        st.header("6. Prediction Results")

        results_df = test_df.copy()

        results_df["Predicted Outcome"] = np.where(
            y_pred == 1,
            "Yes",
            "No"
        )

        results_df["Subscription Probability"] = y_prob

        st.dataframe(
            results_df.head(100),
            use_container_width=True
        )

        # Download predictions

        csv_data = results_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv_data,
            file_name="prediction_results.csv",
            mime="text/csv"
        )

        st.success(
            f"Evaluation completed successfully using {selected_model}!"
        )

    except Exception as e:

        st.error(
            f"An error occurred while processing the dataset: {e}"
        )

else:

    st.info(
        "Please upload your test_data.csv file to begin model evaluation."
    )

# ==================================================
# 10. FOOTER
# ==================================================

st.divider()

st.caption(
    "BITS Pilani | M.Tech AIML | Machine Learning Assignment 2"
)
