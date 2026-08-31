import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Medical Appointment Prediction - Model Performance")


# ==================================================
# LOAD DATASET
# ==================================================

try:
    df = pd.read_csv(
        "data/Medical_appointment_data.csv"
    )

    st.success("✅ Dataset loaded successfully!")

except FileNotFoundError:
    st.error(
        "❌ Dataset not found. Please check:\n"
        "data/Medical_appointment_data.csv"
    )
    st.stop()


# ==================================================
# CLEAN TARGET VARIABLE
# ==================================================

target = "no_show"

if target not in df.columns:
    st.error(
        f"❌ Target column '{target}' is not present in the dataset."
    )
    st.stop()


# Remove missing target values
df = df.dropna(
    subset=[target]
).copy()


# Convert target
# no  = 0 → Show
# yes = 1 → No-Show

y = df[target].astype(str).str.lower().map({
    "no": 0,
    "yes": 1
})


# Remove rows where target conversion failed
valid_rows = y.notna()

df = df.loc[
    valid_rows
].copy()

y = y.loc[
    valid_rows
].astype(int)


# ==================================================
# CREATE FEATURES
# ==================================================

X = df.drop(
    columns=[target]
).copy()


# ==================================================
# FEATURE ENGINEERING
# ==================================================

if "appointment_date_continuous" in X.columns:

    # Convert date column
    X["appointment_date_continuous"] = pd.to_datetime(
        X["appointment_date_continuous"],
        errors="coerce"
    )

    # Extract year
    X["appointment_year"] = (
        X["appointment_date_continuous"].dt.year
    )

    # Extract month
    X["appointment_month"] = (
        X["appointment_date_continuous"].dt.month
    )

    # Extract day
    X["appointment_day"] = (
        X["appointment_date_continuous"].dt.day
    )

    # Extract day of week number
    X["appointment_dayofweek"] = (
        X["appointment_date_continuous"].dt.dayofweek
    )

    # Extract weekday name
    X["appointment_weekday"] = (
        X["appointment_date_continuous"].dt.day_name()
    )


# ==================================================
# DATA INFORMATION
# ==================================================

st.subheader("📋 Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Rows",
        len(X)
    )

with col2:
    st.metric(
        "Total Features",
        X.shape[1]
    )

with col3:
    st.metric(
        "No-Show Patients",
        int(y.sum())
    )


# ==================================================
# TARGET DISTRIBUTION
# ==================================================

st.subheader("🎯 Target Distribution")

target_distribution = pd.DataFrame({
    "Status": [
        "Show",
        "No-Show"
    ],
    "Count": [
        int((y == 0).sum()),
        int((y == 1).sum())
    ]
})

st.dataframe(
    target_distribution,
    use_container_width=True
)


# ==================================================
# SAFETY CHECK
# ==================================================

if len(X) == 0:

    st.error(
        "❌ X contains 0 rows. Please check the dataset."
    )

    st.stop()


if len(y) == 0:

    st.error(
        "❌ Target variable contains 0 rows."
    )

    st.stop()


# ==================================================
# TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


st.success(
    "✅ Train-test split completed successfully!"
)


col1, col2 = st.columns(2)

with col1:
    st.write(
        "Training rows:",
        len(X_train)
    )

with col2:
    st.write(
        "Testing rows:",
        len(X_test)
    )


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

try:

    model = joblib.load(
        "models/logistic_model.pkl"
    )

    st.success(
        "✅ Logistic Regression model loaded!"
    )

except FileNotFoundError:

    st.error(
        "❌ logistic_model.pkl not found.\n\n"
        "Please check the models folder."
    )

    st.stop()


# ==================================================
# LOAD PREPROCESSOR
# ==================================================

try:

    preprocessor = joblib.load(
        "models/preprocessor.pkl"
    )

    st.success(
        "✅ Preprocessor loaded successfully!"
    )

except FileNotFoundError:

    st.error(
        "❌ preprocessor.pkl not found.\n\n"
        "Please check the models folder."
    )

    st.stop()


# ==================================================
# CHECK PREPROCESSOR FEATURES
# ==================================================

st.subheader(
    "🔎 Preprocessor Feature Check"
)


if hasattr(
    preprocessor,
    "feature_names_in_"
):

    expected_columns = list(
        preprocessor.feature_names_in_
    )

else:

    expected_columns = []


current_columns = list(
    X_test.columns
)


# Find missing columns
missing_columns = [
    column
    for column in expected_columns
    if column not in current_columns
]


# Find extra columns
extra_columns = [
    column
    for column in current_columns
    if column not in expected_columns
]


# ==================================================
# HANDLE MISSING COLUMNS
# ==================================================

if missing_columns:

    st.warning(
        "⚠️ Some columns expected by the "
        "preprocessor are missing:"
    )

    st.write(
        missing_columns
    )

    # Create missing columns
    for column in missing_columns:

        # Weekday is categorical
        if column == "appointment_weekday":

            X_test[column] = "Unknown"

        # Other missing columns
        else:

            X_test[column] = 0


# ==================================================
# REMOVE EXTRA COLUMNS
# ==================================================

if expected_columns:

    X_test = X_test[
        expected_columns
    ].copy()


# ==================================================
# PROCESS TEST DATA
# ==================================================

try:

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    st.success(
        "✅ Test data processed successfully!"
    )

except Exception as e:

    st.error(
        "❌ Error while processing test data:"
    )

    st.exception(e)

    st.stop()


st.write(
    "Processed test data shape:",
    X_test_processed.shape
)


# ==================================================
# MODEL PREDICTION
# ==================================================

try:

    y_pred = model.predict(
        X_test_processed
    )

    st.success(
        "✅ Predictions generated successfully!"
    )

except Exception as e:

    st.error(
        "❌ Error while generating predictions:"
    )

    st.exception(e)

    st.stop()


# ==================================================
# CLASSIFICATION METRICS
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ==================================================
# DISPLAY METRICS
# ==================================================

st.subheader(
    "📈 Model Performance Metrics"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )


with col2:

    st.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )


with col3:

    st.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )


with col4:

    st.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )


# ==================================================
# CLASSIFICATION REPORT
# ==================================================

st.subheader(
    "📋 Classification Report"
)


report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Show",
        "No-Show"
    ],
    zero_division=0
)


st.text(report)


# ==================================================
# CONFUSION MATRIX
# ==================================================

st.subheader(
    "🔲 Confusion Matrix"
)


cm = confusion_matrix(
    y_test,
    y_pred
)


fig, ax = plt.subplots(
    figsize=(7, 5)
)


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Show",
        "No-Show"
    ],
    yticklabels=[
        "Show",
        "No-Show"
    ],
    ax=ax
)


ax.set_xlabel(
    "Predicted"
)

ax.set_ylabel(
    "Actual"
)

ax.set_title(
    "Confusion Matrix"
)


st.pyplot(
    fig
)

plt.close(fig)


# ==================================================
# ROC CURVE
# ==================================================

st.subheader(
    "📈 ROC Curve"
)


if hasattr(
    model,
    "predict_proba"
):

    try:

        # Probability of No-Show
        y_probability = model.predict_proba(
            X_test_processed
        )[:, 1]


        # ROC values
        fpr, tpr, thresholds = roc_curve(
            y_test,
            y_probability
        )


        # AUC
        auc_score = roc_auc_score(
            y_test,
            y_probability
        )


        # Plot
        fig, ax = plt.subplots(
            figsize=(8, 6)
        )


        ax.plot(
            fpr,
            tpr,
            label=(
                f"Logistic Regression "
                f"(AUC = {auc_score:.3f})"
            )
        )


        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Random Classifier"
        )


        ax.set_xlabel(
            "False Positive Rate"
        )

        ax.set_ylabel(
            "True Positive Rate"
        )

        ax.set_title(
            "ROC Curve"
        )


        ax.legend()


        st.pyplot(
            fig
        )

        plt.close(fig)


        st.metric(
            "AUC Score",
            f"{auc_score:.3f}"
        )

    except Exception as e:

        st.warning(
            "⚠️ ROC curve could not be generated."
        )

        st.exception(e)

else:

    st.warning(
        "⚠️ This model does not support predict_proba()."
    )


# ==================================================
# FEATURE IMPORTANCE
# ==================================================

st.subheader(
    "🔍 Feature Importance"
)


# Get feature names
try:

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

except Exception:

    feature_names = [
        f"Feature {i}"
        for i in range(
            len(model.coef_[0])
        )
    ]


# Logistic Regression coefficients
if hasattr(
    model,
    "coef_"
):

    coefficients = (
        model.coef_[0]
    )


    # Safety check
    if len(feature_names) == len(
        coefficients
    ):

        feature_importance = pd.DataFrame({

            "Feature":
                feature_names,

            "Coefficient":
                coefficients
        })


        # ==================================================
        # CONVERT ENCODED FEATURES
        # ==================================================

        def get_original_feature(
            feature
        ):

            feature = str(feature)


            if feature.startswith(
                "cat__"
            ):

                remaining = (
                    feature.replace(
                        "cat__",
                        "",
                        1
                    )
                )


                for col in df.columns:

                    if remaining.startswith(
                        col + "_"
                    ):

                        return col


                return remaining


            elif feature.startswith(
                "num__"
            ):

                return feature.replace(
                    "num__",
                    "",
                    1
                )


            else:

                return feature


        # Convert names
        feature_importance[
            "Original Feature"
        ] = (
            feature_importance[
                "Feature"
            ]
            .apply(
                get_original_feature
            )
        )


        # Absolute coefficient
        feature_importance[
            "Importance"
        ] = (
            feature_importance[
                "Coefficient"
            ].abs()
        )


        # Sort
        feature_importance = (
            feature_importance
            .sort_values(
                by="Importance",
                ascending=False
            )
        )


        # ==================================================
        # FEATURE IMPORTANCE TABLE
        # ==================================================

        st.write(
            "Top factors influencing the prediction:"
        )


        st.dataframe(

            feature_importance[
                [
                    "Feature",
                    "Original Feature",
                    "Coefficient",
                    "Importance"
                ]
            ].head(20),

            use_container_width=True
        )


        # ==================================================
        # FEATURE IMPORTANCE CHART
        # ==================================================

        st.subheader(
            "📊 Top 15 Feature Importance"
        )


        top_features = (
            feature_importance
            .head(15)
            .copy()
        )


        fig, ax = plt.subplots(
            figsize=(10, 7)
        )


        ax.barh(
            top_features[
                "Original Feature"
            ],
            top_features[
                "Importance"
            ]
        )


        ax.set_title(
            "Top 15 Feature Importance"
        )


        ax.set_xlabel(
            "Importance"
        )


        ax.set_ylabel(
            "Feature"
        )


        ax.invert_yaxis()


        st.pyplot(
            fig
        )

        plt.close(fig)


    else:

        st.warning(
            "⚠️ Number of feature names "
            "does not match model coefficients."
        )

else:

    st.warning(
        "⚠️ Feature importance is not available "
        "for this model."
    )


# ==================================================
# METRIC EXPLANATION
# ==================================================

st.subheader(
    "📌 Metric Explanation"
)


st.markdown(
    """
### Accuracy
Percentage of total predictions that are correct.

### Precision
Among patients predicted as **No-Show**, how many actually did not show up.

### Recall
Among patients who actually did not show up, how many were correctly identified.

### F1 Score
The harmonic mean of Precision and Recall.

### AUC
Measures how well the model distinguishes between **Show** and **No-Show** appointments.

### Feature Importance
Shows which features have the strongest influence on the Logistic Regression prediction.

**Positive coefficient:** increases the likelihood of No-Show.

**Negative coefficient:** decreases the likelihood of No-Show.
"""
)


# ==================================================
# END
# ==================================================

st.success(
    "🎉 Model Performance Analysis Completed Successfully!"
)

