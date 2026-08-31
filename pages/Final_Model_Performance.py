import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Final Model Performance",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Final Model Performance Dashboard")

st.write(
    """
    This page presents the performance of the final tuned
    Logistic Regression model used for Medical Appointment
    No-Show Prediction.
    """
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "medical_appointments_cleaned.csv"
)

# --------------------------------------------------
# CHECK FILES
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):

    st.error("Final tuned model was not found.")

    st.code(MODEL_PATH)

    st.info(
        "Please run hyperparameter_tuning.py first."
    )

    st.stop()


if not os.path.exists(DATA_PATH):

    st.error("Dataset was not found.")

    st.code(DATA_PATH)

    st.stop()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load(
    MODEL_PATH
)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(
    DATA_PATH
)

# --------------------------------------------------
# CLEAN DATA
# --------------------------------------------------

df.columns = df.columns.str.strip()

df = df.drop_duplicates()

target_column = "no_show"

# Convert target

df[target_column] = (
    df[target_column]
    .astype(str)
    .str.strip()
    .str.lower()
)

df[target_column] = df[target_column].replace({
    "yes": 1,
    "no": 0
})

df[target_column] = pd.to_numeric(
    df[target_column],
    errors="coerce"
)

df = df.dropna(
    subset=[target_column]
)

df[target_column] = (
    df[target_column]
    .astype(int)
)

# --------------------------------------------------
# CREATE X AND y
# --------------------------------------------------

X = df.drop(
    columns=[target_column]
)

y = df[target_column]

# --------------------------------------------------
# APPOINTMENT TIME PROCESSING
# --------------------------------------------------

if "appointment_time" in X.columns:

    X["appointment_time"] = pd.to_datetime(
        X["appointment_time"],
        errors="coerce"
    )

    X["appointment_hour"] = (
        X["appointment_time"].dt.hour
    )

    X["appointment_minute"] = (
        X["appointment_time"].dt.minute
    )

    X = X.drop(
        columns=["appointment_time"]
    )

# --------------------------------------------------
# HANDLE MISSING VALUES
# --------------------------------------------------

numeric_columns = X.select_dtypes(
    include=[
        "int64",
        "float64",
        "int32",
        "float32"
    ]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=[
        "object",
        "category",
        "bool",
        "string"
    ]
).columns.tolist()

for column in numeric_columns:

    X[column] = X[column].fillna(
        X[column].median()
    )

for column in categorical_columns:

    X[column] = X[column].fillna(
        "Unknown"
    )

    X[column] = X[column].astype(str)

# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

try:

    y_pred = model.predict(
        X
    )

    y_prob = model.predict_proba(
        X
    )[:, 1]

except Exception as e:

    st.error(
        "Model prediction failed."
    )

    st.exception(e)

    st.stop()

# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------

accuracy = accuracy_score(
    y,
    y_pred
)

precision = precision_score(
    y,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y,
    y_prob
)

# --------------------------------------------------
# METRICS
# --------------------------------------------------

st.subheader("🏆 Final Model Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )

with col2:

    st.metric(
        "Precision",
        f"{precision:.2%}"
    )

with col3:

    st.metric(
        "Recall",
        f"{recall:.2%}"
    )

with col4:

    st.metric(
        "F1 Score",
        f"{f1:.2%}"
    )

with col5:

    st.metric(
        "ROC-AUC",
        f"{roc_auc:.2%}"
    )

st.divider()

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

st.subheader("🔲 Confusion Matrix")

cm = confusion_matrix(
    y,
    y_pred
)

fig, ax = plt.subplots(
    figsize=(7, 5)
)

ax.imshow(cm)

ax.set_title(
    "Confusion Matrix"
)

ax.set_xlabel(
    "Predicted Label"
)

ax.set_ylabel(
    "Actual Label"
)

ax.set_xticks(
    [0, 1]
)

ax.set_yticks(
    [0, 1]
)

ax.set_xticklabels(
    ["Show", "No-Show"]
)

ax.set_yticklabels(
    ["Show", "No-Show"]
)

for i in range(2):

    for j in range(2):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# ROC CURVE
# --------------------------------------------------

st.subheader("📈 ROC Curve")

fpr, tpr, thresholds = roc_curve(
    y,
    y_prob
)

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

ax.set_xlabel(
    "False Positive Rate"
)

ax.set_ylabel(
    "True Positive Rate"
)

ax.set_title(
    "Receiver Operating Characteristic Curve"
)

ax.legend()

plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# PRECISION-RECALL CURVE
# --------------------------------------------------

st.subheader("📊 Precision-Recall Curve")

precision_values, recall_values, pr_thresholds = (
    precision_recall_curve(
        y,
        y_prob
    )
)

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    recall_values,
    precision_values
)

ax.set_xlabel(
    "Recall"
)

ax.set_ylabel(
    "Precision"
)

ax.set_title(
    "Precision-Recall Curve"
)

plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

st.subheader(
    "📋 Classification Report"
)

report = classification_report(
    y,
    y_pred,
    target_names=[
        "Show",
        "No-Show"
    ],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.subheader(
    "🤖 Final Model Information"
)

st.write(
    "Model: Tuned Logistic Regression"
)

st.write(
    "Model file: models/final_model.pkl"
)

st.write(
    "Objective: Medical Appointment No-Show Prediction"
)

st.info(
    """
    The final model was selected after hyperparameter
    tuning. The tuned configuration uses class_weight='balanced'
    to improve detection of the No-Show class.
    """
)

# --------------------------------------------------
# FINAL CONCLUSION
# --------------------------------------------------

st.subheader(
    "📝 Final Conclusion"
)

st.success(
    f"""
    The final tuned Logistic Regression model achieved:

    Accuracy: {accuracy:.2%}

    Precision: {precision:.2%}

    Recall: {recall:.2%}

    F1 Score: {f1:.2%}

    ROC-AUC: {roc_auc:.2%}

    The model is particularly effective at identifying
    patients who are likely to miss their appointments,
    as reflected by its No-Show recall.
    """
)

