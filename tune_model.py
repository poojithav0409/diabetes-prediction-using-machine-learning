
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from data_utils import load_data, clean_data


# ---------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------

df = load_data()
df = clean_data(df)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# ---------------------------------------
# STEP 2: SPLIT DATA
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------
# STEP 3: CREATE PIPELINES
# ---------------------------------------

logistic_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])


rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        random_state=42
    ))
])


# ---------------------------------------
# STEP 4: DEFINE PARAMETERS
# ---------------------------------------

logistic_params = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__class_weight": [None, "balanced"]
}

rf_params = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 5, 10],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}


# ---------------------------------------
# STEP 5: CROSS VALIDATION
# ---------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ---------------------------------------
# STEP 6: HYPERPARAMETER TUNING
# ---------------------------------------

searches = {
    "Logistic Regression": GridSearchCV(
        logistic_pipeline,
        logistic_params,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    ),

    "Random Forest": GridSearchCV(
        rf_pipeline,
        rf_params,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )
}


best_models = {}
results = {}

for name, search in searches.items():

    print("\nTuning:", name)

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    best_models[name] = best_model

    print("Best Parameters:")
    print(search.best_params_)

    print("Best Cross-validation F1-score:")
    print(round(search.best_score_, 4))

    # Evaluate on untouched test data
    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results[name] = {
        "Accuracy": round(float(accuracy), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1-score": round(float(f1), 4)
    }

    print("\nFinal Test Results:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))


# ---------------------------------------
# STEP 7: SAVE RESULTS
# ---------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

joblib.dump(
    best_models,
    "models/diabetes_models.pkl"
)

with open("outputs/tuned_model_metrics.json", "w") as file:
    json.dump(results, file, indent=4)

pd.DataFrame(results).T.to_csv(
    "outputs/tuned_model_comparison.csv"
)

print("\nMODEL TUNING COMPLETED!")
