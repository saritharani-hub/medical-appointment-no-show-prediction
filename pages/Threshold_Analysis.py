import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Threshold Analysis",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Prediction Threshold Analysis")

st.write(
    """
    This page evaluates different probability thresholds
    for detecting medical appointment No-Shows.

    The default Logistic Regression threshold is 0.50.
    We compare several thresholds to determine which gives
    the best balance between Precision and Recall.
    """
)

# --------------------------------------------------
# PROJECT PATH
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
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(
    DATA_PATH
)

st.success(
    "Model and dataset loaded successfully."
)

# --------------------------------------------------
# CLEAN DATA
# --------------------------------------------------

df.columns = df.columns.str.strip()

df = df.drop_duplicates()

target_column = "no_show"

# Convert target to numeric

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

df[target_column] = df[target_column].astype(int)

# --------------------------------------------------
# CREATE X AND y
# --------------------------------------------------

X = df.drop(
    columns=[target_column]
)

y = df[target_column]

# --------------------------------------------------
# APPOINTMENT TIME
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
# PREDICT PROBABILITIES
# --------------------------------------------------

st.info(
    "Generating No-Show probabilities..."
)

try:

    y_prob = model.predict_proba(
        X
    )[:, 1]

except Exception as e:

    st.error(
        "Prediction failed because the input "
        "features do not match the trained model."
    )

    st.exception(e)

    st.stop()

# --------------------------------------------------
# THRESHOLD ANALYSIS
# --------------------------------------------------

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)

results = []

for threshold in thresholds:
    y_pred_threshold = (
            y_prob >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y,
        y_pred_threshold
    )

    precision = precision_score(
        y,
        y_pred_threshold,
        zero_division=0
    )

    recall = recall_score(
        y,
        y_pred_threshold,
        zero_division=0
    )

    f1 = f1_score(
        y,
        y_pred_threshold,
        zero_division=0
    )

    results.append({

        "Threshold": round(
            threshold,
            2
        ),

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1
    })

results_df = pd.DataFrame(
    results
)

# --------------------------------------------------
# FIND BEST THRESHOLD
# --------------------------------------------------

best_row = results_df.loc[
    results_df["F1 Score"].idxmax()
]

best_threshold = best_row[
    "Threshold"
]

best_f1 = best_row[
    "F1 Score"
]

# --------------------------------------------------
# DISPLAY BEST THRESHOLD
# --------------------------------------------------

st.subheader(
    "🏆 Best Threshold"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Best Threshold",
        f"{best_threshold:.2f}"
    )

with col2:
    st.metric(
        "Best F1 Score",
        f"{best_f1:.4f}"
    )

with col3:
    st.metric(
        "Default Threshold",
        "0.50"
    )

# --------------------------------------------------
# RESULTS TABLE
# --------------------------------------------------

st.subheader(
    "📊 Threshold Comparison"
)

st.dataframe(
    results_df.round(4),
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# CHART
# --------------------------------------------------

st.subheader(
    "📈 Threshold vs Model Metrics"
)

chart_df = results_df.set_index(
    "Threshold"
)[
    [
        "Precision",
        "Recall",
        "F1 Score"
    ]
]

st.line_chart(
    chart_df
)

# --------------------------------------------------
# DEFAULT THRESHOLD RESULT
# --------------------------------------------------

default_row = results_df[
    results_df["Threshold"] == 0.50
    ]

if not default_row.empty:
    default_f1 = default_row[
        "F1 Score"
    ].iloc[0]

    default_recall = default_row[
        "Recall"
    ].iloc[0]

    default_precision = default_row[
        "Precision"
    ].iloc[0]

    st.subheader(
        "🔎 Default Threshold (0.50)"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Precision",
            f"{default_precision:.4f}"
        )

    with col2:
        st.metric(
            "Recall",
            f"{default_recall:.4f}"
        )

    with col3:
        st.metric(
            "F1 Score",
            f"{default_f1:.4f}"
        )

# --------------------------------------------------
# BEST THRESHOLD DETAILS
# --------------------------------------------------

best_precision = best_row[
    "Precision"
]

best_recall = best_row[
    "Recall"
]

best_accuracy = best_row[
    "Accuracy"
]

st.subheader(
    "🎯 Best Threshold Performance"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy",
        f"{best_accuracy:.4f}"
    )

with col2:
    st.metric(
        "Precision",
        f"{best_precision:.4f}"
    )

with col3:
    st.metric(
        "Recall",
        f"{best_recall:.4f}"
    )

with col4:
    st.metric(
        "F1 Score",
        f"{best_f1:.4f}"
    )

# --------------------------------------------------
# EXPLANATION
# --------------------------------------------------

st.subheader(
    "ℹ️ Interpretation"
)

st.info(
    f"""
    The threshold determines when the model predicts
    No-Show.

    Current default threshold: 0.50

    Best threshold based on F1 Score: {best_threshold:.2f}

    A lower threshold generally identifies more potential
    No-Show patients, increasing Recall but potentially
    reducing Precision.

    A higher threshold generally produces fewer No-Show
    predictions, which can increase Precision but reduce
    Recall.

    The selected threshold should ultimately depend on the
    practical cost of missed appointments versus unnecessary
    reminders/interventions.
    """
)

