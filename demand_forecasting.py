import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# DEMAND FORECASTING - STEP 1
# ============================================================

# Load dataset
df = pd.read_csv("data/Medical_appointment_data.csv")

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

# Convert date
df["appointment_date_continuous"] = pd.to_datetime(
    df["appointment_date_continuous"],
    errors="coerce"
)

# ============================================================
# CREATE DAILY DEMAND
# ============================================================

daily_demand = (
    df.groupby("appointment_date_continuous")
      .size()
      .reset_index(name="appointment_count")
)

print("\nDaily Demand:")
print(daily_demand.head())

print("\nDaily demand shape:", daily_demand.shape)

# Save daily demand
daily_demand.to_csv(
    "daily_appointment_demand.csv",
    index=False
)

print("Saved: daily_appointment_demand.csv")

# ============================================================
# STEP 2 - VISUALIZE DAILY DEMAND
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    daily_demand["appointment_date_continuous"],
    daily_demand["appointment_count"]
)

plt.title("Daily Medical Appointment Demand")
plt.xlabel("Appointment Date")
plt.ylabel("Number of Appointments")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("images/daily_appointment_demand.png")
plt.show()

print("Saved: images/daily_appointment_demand.png")

# ============================================================
# STEP 3 - PREPARE TIME SERIES DATA
# ============================================================

# Create time-based features
daily_demand["year"] = daily_demand["appointment_date_continuous"].dt.year
daily_demand["month"] = daily_demand["appointment_date_continuous"].dt.month
daily_demand["day"] = daily_demand["appointment_date_continuous"].dt.day
daily_demand["day_of_week"] = (
    daily_demand["appointment_date_continuous"].dt.dayofweek
)

# Create lag features
daily_demand["lag_1"] = daily_demand["appointment_count"].shift(1)
daily_demand["lag_7"] = daily_demand["appointment_count"].shift(7)

# Rolling average
daily_demand["rolling_7"] = (
    daily_demand["appointment_count"]
    .shift(1)
    .rolling(7)
    .mean()
)

# Remove rows created by lag features
daily_demand = daily_demand.dropna().reset_index(drop=True)

print("\nPrepared forecasting data:")
print(daily_demand.head(10))

print("\nPrepared data shape:", daily_demand.shape)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(daily_demand) * 0.80)

train_data = daily_demand.iloc[:split_index]
test_data = daily_demand.iloc[split_index:]

print("\nTraining data:", train_data.shape)
print("Testing data:", test_data.shape)

print("\nTraining period:")
print(
    train_data["appointment_date_continuous"].min(),
    "to",
    train_data["appointment_date_continuous"].max()
)

print("\nTesting period:")
print(
    test_data["appointment_date_continuous"].min(),
    "to",
    test_data["appointment_date_continuous"].max()
)

# ============================================================
# STEP 4 - DEMAND FORECASTING MODELS
# ============================================================

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ------------------------------------------------------------
# Features and target
# ------------------------------------------------------------

features = [
    "year",
    "month",
    "day",
    "day_of_week",
    "lag_1",
    "lag_7",
    "rolling_7"
]

X_train = train_data[features]
y_train = train_data["appointment_count"]

X_test = test_data[features]
y_test = test_data["appointment_count"]

# ------------------------------------------------------------
# Create models
# ------------------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

# ------------------------------------------------------------
# Train and evaluate models
# ------------------------------------------------------------

results = []

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

# ------------------------------------------------------------
# Model comparison
# ------------------------------------------------------------

model_results = pd.DataFrame(results)

print("\n============================================================")
print("MODEL COMPARISON")
print("============================================================")

print(model_results)

# Save results
model_results.to_csv(
    "models/demand_forecasting_model_comparison.csv",
    index=False
)

print("\nSaved: models/demand_forecasting_model_comparison.csv")

# ============================================================
# STEP 5 - BEST MODEL AND FORECAST VISUALIZATION
# ============================================================

import joblib

# ------------------------------------------------------------
# Train the selected Random Forest model again
# ------------------------------------------------------------

best_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

best_model.fit(X_train, y_train)

# Generate predictions
test_predictions = best_model.predict(X_test)

# ------------------------------------------------------------
# Create forecast results table
# ------------------------------------------------------------

forecast_results = pd.DataFrame({
    "appointment_date": test_data["appointment_date_continuous"],
    "actual_demand": y_test.values,
    "forecasted_demand": test_predictions
})

print("\n============================================================")
print("FORECAST RESULTS")
print("============================================================")

print(forecast_results.head(10))

# Save forecast results
forecast_results.to_csv(
    "models/demand_forecast_results.csv",
    index=False
)

print("\nSaved: models/demand_forecast_results.csv")

# ------------------------------------------------------------
# Save Random Forest model
# ------------------------------------------------------------

joblib.dump(
    best_model,
    "models/demand_forecasting_model.pkl"
)

print("Saved: models/demand_forecasting_model.pkl")

# ------------------------------------------------------------
# Actual vs Forecast graph
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    forecast_results["appointment_date"],
    forecast_results["actual_demand"],
    label="Actual Demand"
)

plt.plot(
    forecast_results["appointment_date"],
    forecast_results["forecasted_demand"],
    label="Forecasted Demand"
)

plt.title("Actual vs Forecasted Medical Appointment Demand")
plt.xlabel("Appointment Date")
plt.ylabel("Number of Appointments")
plt.legend()

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "images/actual_vs_forecasted_demand.png"
)

plt.show()

print("Saved: images/actual_vs_forecasted_demand.png")