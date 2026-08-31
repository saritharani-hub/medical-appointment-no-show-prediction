# ============================================================
# HYPERPARAMETER TUNING
# MEDICAL APPOINTMENT NO-SHOW PREDICTION
# ============================================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/medical_appointments_cleaned.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)

df = df.drop_duplicates()

print("Shape after duplicates:", df.shape)


# ============================================================
# 4. TARGET VARIABLE
# ============================================================

target_column = "no_show"

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


print("\nTarget distribution:")
print(df[target_column].value_counts())


# ============================================================
# 5. CREATE X AND y
# ============================================================

X = df.drop(
    columns=[target_column]
)

y = df[target_column]


# ============================================================
# 6. HANDLE APPOINTMENT TIME
# ============================================================

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


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

numeric_columns = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool", "string"]
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


# ============================================================
# 8. TRAIN TEST SPLIT
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
# 9. IDENTIFY FEATURES
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool", "string"]
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
# 11. LOGISTIC REGRESSION PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# ============================================================
# 12. HYPERPARAMETER GRID
# ============================================================

param_grid = {
    "classifier__C": [
        0.1,
        1,
        10
    ],

    "classifier__class_weight": [
        None,
        "balanced"
    ]
}


# ============================================================
# 13. GRID SEARCH
# ============================================================

print("\nStarting Hyperparameter Tuning...")

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=2,
    n_jobs=-1,
    verbose=2
)


grid_search.fit(
    X_train,
    y_train
)


# ============================================================
# 14. BEST PARAMETERS
# ============================================================

print("\n================================================")
print("BEST PARAMETERS")
print("================================================")

print(
    grid_search.best_params_
)


print("\nBest Cross-Validation F1 Score:")

print(
    round(
        grid_search.best_score_,
        4
    )
)


# ============================================================
# 15. BEST MODEL
# ============================================================

best_model = grid_search.best_estimator_


# ============================================================
# 16. TEST SET PREDICTION
# ============================================================

y_pred = best_model.predict(
    X_test
)

y_prob = best_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 17. EVALUATION
# ============================================================

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


print("\n================================================")
print("TUNED MODEL PERFORMANCE")
print("================================================")

print(
    "Accuracy :", round(accuracy, 4)
)

print(
    "Precision:", round(precision, 4)
)

print(
    "Recall   :", round(recall, 4)
)

print(
    "F1 Score :", round(f1, 4)
)

print(
    "ROC-AUC  :", round(roc_auc, 4)
)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Show",
            "No-Show"
        ],
        zero_division=0
    )
)


# ============================================================
# 19. SAVE FINAL MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    best_model,
    "models/final_model.pkl"
)

print(
    "\nFinal tuned model saved successfully!"
)

print(
    "Location: models/final_model.pkl"
)