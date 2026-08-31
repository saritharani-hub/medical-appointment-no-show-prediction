# ============================================================
# MEDICAL APPOINTMENT NO-SHOW PREDICTION
# MODEL TRAINING + MODEL COMPARISON
# ============================================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "data/medical_appointments_cleaned.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

print("\nDuplicate rows:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# ============================================================
# 4. HANDLE TARGET COLUMN
# ============================================================

target_column = "no_show"

if target_column not in df.columns:
    raise KeyError(
        f"Target column '{target_column}' was not found. "
        f"Available columns are: {df.columns.tolist()}"
    )

# ------------------------------------------------------------
# Convert no_show values
# yes = 1  -> No Show
# no  = 0  -> Show
# ------------------------------------------------------------

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

# Convert to numeric
df[target_column] = pd.to_numeric(
    df[target_column],
    errors="coerce"
)

# Remove invalid target values
df = df.dropna(
    subset=[target_column]
)

# Convert target to integer
df[target_column] = df[target_column].astype(int)

print("\nTarget distribution:")
print(df[target_column].value_counts())

print("\nTarget values:")
print(df[target_column].unique())

# ============================================================
# 5. REMOVE UNNECESSARY COLUMNS
# ============================================================

# These columns should NOT be used as prediction features
columns_to_drop = [
    target_column
]

# Drop ID-like columns if they exist
possible_id_columns = [
    "patientid",
    "patient_id",
    "appointmentid",
    "appointment_id"
]

for col in possible_id_columns:
    if col in df.columns:
        columns_to_drop.append(col)

X = df.drop(columns=columns_to_drop)
y = df[target_column]


# ============================================================
# 6. HANDLE DATE/TIME COLUMNS
# ============================================================

# Convert appointment_time to useful numeric information
if "appointment_time" in X.columns:

    X["appointment_time"] = pd.to_datetime(
        X["appointment_time"],
        errors="coerce"
    )

    X["appointment_hour"] = X["appointment_time"].dt.hour

    X["appointment_minute"] = X["appointment_time"].dt.minute

    X = X.drop(columns=["appointment_time"])


# Convert appointment_date if present
if "appointment_date" in X.columns:

    X["appointment_date"] = pd.to_datetime(
        X["appointment_date"],
        errors="coerce"
    )

    X["appointment_year"] = X["appointment_date"].dt.year

    X["appointment_month"] = X["appointment_date"].dt.month

    X["appointment_day"] = X["appointment_date"].dt.day

    X["appointment_dayofweek"] = (
        X["appointment_date"].dt.dayofweek
    )

    X = X.drop(columns=["appointment_date"])


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

# Numeric columns
numeric_columns = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

# Categorical columns
categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# Fill numeric missing values
for col in numeric_columns:
    X[col] = X[col].fillna(X[col].median())


# Fill categorical missing values
for col in categorical_columns:
    X[col] = X[col].fillna("Unknown")
    X[col] = X[col].astype(str)


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 9. IDENTIFY COLUMNS AGAIN
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# ============================================================
# 10. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 11. DEFINE MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

}


# ============================================================
# 12. MODEL COMPARISON
# ============================================================

results = []

trained_models = {}


for name, classifier in models.items():

    print("\n----------------------------------------")
    print("Training:", name)
    print("----------------------------------------")

    # Create pipeline
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),

            (
                "classifier",
                classifier
            )
        ]
    )

    # Train
    pipeline.fit(
        X_train,
        y_train
    )

    # Prediction
    y_pred = pipeline.predict(X_test)

    # Probability
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
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

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    # Store results
    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": roc_auc

    })

    # Store trained model
    trained_models[name] = pipeline

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))


# ============================================================
# 13. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# 14. SORT BY F1 SCORE
# ============================================================

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 15. DISPLAY MODEL COMPARISON
# ============================================================

print("\n\n================================================")
print("MODEL COMPARISON")
print("================================================")

print(
    results_df.round(4).to_string(index=False)
)


# ============================================================
# 16. FIND BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

print("\n================================================")
print("BEST MODEL")
print("================================================")

print("Best Model:", best_model_name)


# ============================================================
# 17. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# 18. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/model.pkl"
)

print("\nBest model saved successfully!")
print("Location: models/model.pkl")


# ============================================================
# 19. SAVE MODEL COMPARISON RESULTS
# ============================================================

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print(
    "Comparison results saved to: "
    "models/model_comparison.csv"
)


# ============================================================
# 20. SAVE TEST DATA INFORMATION
# ============================================================

print("\nTraining completed successfully!")