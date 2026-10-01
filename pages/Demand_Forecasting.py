import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Medical Appointment Demand Forecasting")

st.write(
    "This dashboard analyzes historical medical appointment demand "
    "and presents machine learning forecasting results."
)

# ============================================================
# FILE PATHS
# ============================================================

forecast_file = "models/demand_forecast_results.csv"
comparison_file = "models/demand_forecasting_model_comparison.csv"
daily_file = "daily_appointment_demand.csv"
model_file = "models/demand_forecasting_model.pkl"

# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    forecast_file,
    comparison_file,
    daily_file,
    model_file
]

for file in required_files:
    if not os.path.exists(file):
        st.error(f"Required file not found: {file}")
        st.stop()

# ============================================================
# LOAD DATA
# ============================================================

forecast_df = pd.read_csv(forecast_file)

comparison_df = pd.read_csv(
    comparison_file
)

daily_df = pd.read_csv(
    daily_file
)

# Load Random Forest model
model = joblib.load(model_file)

# Convert dates
forecast_df["appointment_date"] = pd.to_datetime(
    forecast_df["appointment_date"]
)

daily_df["appointment_date_continuous"] = pd.to_datetime(
    daily_df["appointment_date_continuous"]
)

# ============================================================
# SECTION 1 - OVERVIEW METRICS
# ============================================================

st.subheader("📊 Forecasting Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Daily Records",
        len(daily_df)
    )

with col2:
    st.metric(
        "Test Records",
        len(forecast_df)
    )

with col3:
    st.metric(
        "Average Daily Demand",
        round(
            daily_df["appointment_count"].mean(),
            2
        )
    )

with col4:
    st.metric(
        "Maximum Daily Demand",
        int(
            daily_df["appointment_count"].max()
        )
    )

# ============================================================
# SECTION 2 - DAILY DEMAND
# ============================================================

st.subheader("📅 Daily Appointment Demand")

fig1, ax1 = plt.subplots(
    figsize=(14, 5)
)

ax1.plot(
    daily_df["appointment_date_continuous"],
    daily_df["appointment_count"]
)

ax1.set_title(
    "Daily Medical Appointment Demand"
)

ax1.set_xlabel(
    "Appointment Date"
)

ax1.set_ylabel(
    "Number of Appointments"
)

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig1)

# ============================================================
# SECTION 3 - MODEL PERFORMANCE
# ============================================================

st.subheader("🤖 Forecasting Model Performance")

# Get Random Forest row
rf_result = comparison_df[
    comparison_df["Model"] == "Random Forest"
]

if not rf_result.empty:

    rf_mae = rf_result.iloc[0]["MAE"]
    rf_rmse = rf_result.iloc[0]["RMSE"]
    rf_r2 = rf_result.iloc[0]["R2 Score"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Random Forest MAE",
            f"{rf_mae:.2f}"
        )

    with col2:
        st.metric(
            "Random Forest RMSE",
            f"{rf_rmse:.2f}"
        )

    with col3:
        st.metric(
            "Random Forest R²",
            f"{rf_r2:.4f}"
        )

st.write("### Model Comparison")

st.dataframe(
    comparison_df,
    use_container_width=True
)

# ============================================================
# SECTION 4 - ACTUAL VS FORECAST
# ============================================================

st.subheader("📈 Actual vs Forecasted Demand")

fig2, ax2 = plt.subplots(
    figsize=(14, 5)
)

ax2.plot(
    forecast_df["appointment_date"],
    forecast_df["actual_demand"],
    label="Actual Demand"
)

ax2.plot(
    forecast_df["appointment_date"],
    forecast_df["forecasted_demand"],
    label="Forecasted Demand"
)

ax2.set_title(
    "Actual vs Forecasted Medical Appointment Demand"
)

ax2.set_xlabel(
    "Appointment Date"
)

ax2.set_ylabel(
    "Number of Appointments"
)

ax2.legend()

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig2)

# ============================================================
# SECTION 5 - FEATURE IMPORTANCE
# ============================================================

st.subheader("🌳 Forecasting Feature Importance")

feature_names = [
    "year",
    "month",
    "day",
    "day_of_week",
    "lag_1",
    "lag_7",
    "rolling_7"
]

if hasattr(model, "feature_importances_"):

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    # Display table
    st.dataframe(
        importance_df,
        use_container_width=True
    )

    # Feature importance graph
    fig3, ax3 = plt.subplots(
        figsize=(10, 5)
    )

    ax3.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax3.set_title(
        "Forecasting Feature Importance"
    )

    ax3.set_xlabel(
        "Importance"
    )

    ax3.set_ylabel(
        "Feature"
    )

    ax3.invert_yaxis()

    plt.tight_layout()

    st.pyplot(fig3)

else:

    st.warning(
        "Feature importance is not available for this model."
    )

# ============================================================
# SECTION 6 - FORECAST RESULTS
# ============================================================

st.subheader("📋 Forecast Results")

st.dataframe(
    forecast_df,
    use_container_width=True
)

# ============================================================
# SECTION 7 - DOWNLOAD
# ============================================================

st.subheader("⬇️ Download Forecast Results")

csv_data = forecast_df.to_csv(
    index=False
)

st.download_button(
    label="Download Forecast Results CSV",
    data=csv_data,
    file_name="demand_forecast_results.csv",
    mime="text/csv"
)

# ============================================================
# FINAL MESSAGE
# ============================================================

st.success(
    "Demand Forecasting dashboard loaded successfully."
)