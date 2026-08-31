import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Feature Importance",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Feature Importance")
st.write(
    "This page shows which patient, appointment, medical, "
    "and weather features have the strongest influence on "
    "the no-show prediction."
)

# --------------------------------------------------
# LOAD FINAL MODEL
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_model.pkl"
)

if not os.path.exists(MODEL_PATH):
    st.error("Final tuned model was not found.")
    st.code(MODEL_PATH)
    st.info("Please run hyperparameter_tuning.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# CHECK PIPELINE
# --------------------------------------------------

try:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

except Exception:
    st.error(
        "The saved model is not in the expected Pipeline format."
    )
    st.stop()

# --------------------------------------------------
# GET FEATURE NAMES
# --------------------------------------------------

try:

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

except Exception as e:

    st.error("Unable to extract feature names.")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# GET COEFFICIENTS
# --------------------------------------------------

coefficients = classifier.coef_[0]

# --------------------------------------------------
# CREATE FEATURE IMPORTANCE DATAFRAME
# --------------------------------------------------

feature_importance = pd.DataFrame({

    "Feature": feature_names,

    "Coefficient": coefficients,

    "Absolute Importance": abs(coefficients)
})

# --------------------------------------------------
# SORT FEATURES
# --------------------------------------------------

feature_importance = feature_importance.sort_values(
    by="Absolute Importance",
    ascending=False
)

# --------------------------------------------------
# DISPLAY ALL FEATURES
# --------------------------------------------------

st.subheader("🔍 Feature Importance Table")

st.dataframe(
    feature_importance.round(4),
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# TOP 15 FEATURES
# --------------------------------------------------

top_features = (
    feature_importance
    .head(15)
    .sort_values(
        by="Coefficient"
    )
)

st.subheader("📈 Top 15 Important Features")

fig, ax = plt.subplots(
    figsize=(10, 7)
)

ax.barh(
    top_features["Feature"],
    top_features["Coefficient"]
)

ax.axvline(
    x=0,
    linewidth=1
)

ax.set_xlabel(
    "Logistic Regression Coefficient"
)

ax.set_ylabel(
    "Feature"
)

ax.set_title(
    "Top 15 Features Influencing No-Show Prediction"
)

plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# POSITIVE FEATURES
# --------------------------------------------------

st.subheader("🔴 Features Increasing No-Show Likelihood")

positive_features = (
    feature_importance[
        feature_importance["Coefficient"] > 0
    ]
    .head(10)
)

st.dataframe(
    positive_features.round(4),
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# NEGATIVE FEATURES
# --------------------------------------------------

st.subheader("🟢 Features Increasing Show Likelihood")

negative_features = (
    feature_importance[
        feature_importance["Coefficient"] < 0
    ]
    .sort_values(
        by="Coefficient",
        ascending=True
    )
    .head(10)
)

st.dataframe(
    negative_features.round(4),
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# EXPLANATION
# --------------------------------------------------

st.subheader("ℹ️ How to Interpret the Coefficients")

st.info(
    """
    Positive coefficient → increases the model's tendency
    toward the No-Show class (1).

    Negative coefficient → increases the model's tendency
    toward the Show class (0).

    Larger absolute coefficient → stronger influence on
    the model's prediction.

    Important: feature importance shows association used by
    the model; it does not prove that a feature causes a
    patient to miss an appointment.
    """
)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Feature Importance",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Feature Importance")
st.write(
    "This page shows which patient, appointment, medical, "
    "and weather features have the strongest influence on "
    "the no-show prediction."
)

# --------------------------------------------------
# LOAD FINAL MODEL
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_model.pkl"
)

if not os.path.exists(MODEL_PATH):
    st.error("Final tuned model was not found.")
    st.code(MODEL_PATH)
    st.info("Please run hyperparameter_tuning.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# CHECK PIPELINE
# --------------------------------------------------

try:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

except Exception:
    st.error(
        "The saved model is not in the expected Pipeline format."
    )
    st.stop()

# --------------------------------------------------
# GET FEATURE NAMES
# --------------------------------------------------

try:

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

except Exception as e:

    st.error("Unable to extract feature names.")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# GET COEFFICIENTS
# --------------------------------------------------

coefficients = classifier.coef_[0]

# --------------------------------------------------
# CREATE FEATURE IMPORTANCE DATAFRAME
# --------------------------------------------------

feature_importance = pd.DataFrame({

    "Feature": feature_names,

    "Coefficient": coefficients,

    "Absolute Importance": abs(coefficients)
})

# --------------------------------------------------
# SORT FEATURES
# --------------------------------------------------

feature_importance = feature_importance.sort_values(
    by="Absolute Importance",
    ascending=False
)

# --------------------------------------------------
# DISPLAY ALL FEATURES
# --------------------------------------------------

st.subheader("🔍 Feature Importance Table")

st.dataframe(
    feature_importance.round(4),
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# TOP 15 FEATURES
# --------------------------------------------------

top_features = (
    feature_importance
    .head(15)
    .sort_values(
        by="Coefficient"
    )
)

st.subheader("📈 Top 15 Important Features")

fig, ax = plt.subplots(
    figsize=(10, 7)
)

ax.barh(
    top_features["Feature"],
    top_features["Coefficient"]
)

ax.axvline(
    x=0,
    linewidth=1
)

ax.set_xlabel(
    "Logistic Regression Coefficient"
)

ax.set_ylabel(
    "Feature"
)

ax.set_title(
    "Top 15 Features Influencing No-Show Prediction"
)

plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# POSITIVE FEATURES
# --------------------------------------------------

st.subheader("🔴 Features Increasing No-Show Likelihood")

positive_features = (
    feature_importance[
        feature_importance["Coefficient"] > 0
    ]
    .head(10)
)

st.dataframe(
    positive_features.round(4),
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# NEGATIVE FEATURES
# --------------------------------------------------

st.subheader("🟢 Features Increasing Show Likelihood")

negative_features = (
    feature_importance[
        feature_importance["Coefficient"] < 0
    ]
    .sort_values(
        by="Coefficient",
        ascending=True
    )
    .head(10)
)

st.dataframe(
    negative_features.round(4),
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# EXPLANATION
# --------------------------------------------------

st.subheader("ℹ️ How to Interpret the Coefficients")

st.info(
    """
    Positive coefficient → increases the model's tendency
    toward the No-Show class (1).

    Negative coefficient → increases the model's tendency
    toward the Show class (0).

    Larger absolute coefficient → stronger influence on
    the model's prediction.

    Important: feature importance shows association used by
    the model; it does not prove that a feature causes a
    patient to miss an appointment.
    """
)

