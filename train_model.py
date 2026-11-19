
import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
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
    confusion_matrix,
    classification_report
)

from data_utils import load_data, clean_data


# ------------------------------------------------
# STEP 1: Create folders
# ------------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ------------------------------------------------
# STEP 2: Load dataset
# ------------------------------------------------

df = load_data()

df = clean_data(df)

print("Dataset loaded successfully!")


# ------------------------------------------------
# STEP 3: Separate features and target
# ------------------------------------------------

X = df.drop("Outcome", axis=1)

y = df["Outcome"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())


# ------------------------------------------------
# STEP 4: Split training and testing data
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ------------------------------------------------
# STEP 5: Create machine learning models
# ------------------------------------------------

logistic_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])


random_forest_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])


# ------------------------------------------------
# STEP 6: Train both models
# ------------------------------------------------

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model
}

results = {}

for model_name, model in models.items():

    print("\n" + "=" * 50)
    print("Training:", model_name)
    print("=" * 50)

    # Train the model
    model.fit(X_train, y_train)

    # Predict using test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    # Store results
    results[model_name] = {
        "Accuracy": round(float(accuracy), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1-score": round(float(f1), 4)
    }

    # Display evaluation metrics
    print("\nResults for", model_name)

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

    # Classification report
    print("\nClassification Report:")

    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))


    # ------------------------------------------------
    # STEP 7: Confusion Matrix Analysis
    # ------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix -", model_name)

    print(cm)

    # Extract TN, FP, FN and TP
    tn, fp, fn, tp = cm.ravel()

    print("\nDetailed Confusion Matrix Analysis:")

    print("True Negative (TN):", tn)

    print("False Positive (FP):", fp)

    print("False Negative (FN):", fn)

    print("True Positive (TP):", tp)

    print("\nInterpretation:")

    print(
        "Correctly predicted negative cases:",
        tn
    )

    print(
        "Incorrectly predicted positive cases:",
        fp
    )

    print(
        "Missed positive cases:",
        fn
    )

    print(
        "Correctly predicted positive cases:",
        tp
    )


    # ------------------------------------------------
    # STEP 8: Plot Confusion Matrix
    # ------------------------------------------------

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative (0)", "Positive (1)"],
        yticklabels=["Negative (0)", "Positive (1)"]
    )

    plt.title(f"{model_name} - Confusion Matrix")

    plt.xlabel("Predicted Outcome")

    plt.ylabel("Actual Outcome")

    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")

    plt.savefig(
        f"outputs/{safe_name}_confusion_matrix.png"
    )

    plt.close()

    print("\nConfusion matrix saved successfully!")


# ------------------------------------------------
# STEP 9: Compare both models
# ------------------------------------------------

results_df = pd.DataFrame(results).T

print("\n" + "=" * 50)

print("MODEL COMPARISON")

print("=" * 50)

print(results_df)

results_df.to_csv(
    "outputs/model_comparison.csv"
)


# ------------------------------------------------
# STEP 10: Save trained models
# ------------------------------------------------

joblib.dump(
    models,
    "models/diabetes_models.pkl"
)

print("\nModels saved successfully!")


# ------------------------------------------------
# STEP 11: Save evaluation results
# ------------------------------------------------

with open("outputs/model_metrics.json", "w") as file:

    json.dump(results, file, indent=4)

print("Model metrics saved successfully!")


# ------------------------------------------------
# STEP 12: Random Forest Feature Importance
# ------------------------------------------------

rf = models["Random Forest"]

importance = rf.named_steps["classifier"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nRandom Forest Feature Importance:")

print(importance_df)


# Plot feature importance

plt.figure(figsize=(9, 6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest Feature Importance")

plt.xlabel("Importance Score")

plt.ylabel("Patient Measurements")

plt.tight_layout()

plt.savefig("outputs/feature_importance.png")

plt.close()


# ------------------------------------------------
# STEP 13: Final message
# ------------------------------------------------

print("\n" + "=" * 50)

print("MODEL TRAINING COMPLETED SUCCESSFULLY!")

print("=" * 50)

print("\nGenerated Files:")

print("1. models/diabetes_models.pkl")

print("2. outputs/model_metrics.json")

print("3. outputs/model_comparison.csv")

print("4. outputs/logistic_regression_confusion_matrix.png")

print("5. outputs/random_forest_confusion_matrix.png")

print("6. outputs/feature_importance.png")
