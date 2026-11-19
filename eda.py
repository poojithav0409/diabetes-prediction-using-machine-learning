
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import load_data, clean_data

# Load and clean the dataset
df = load_data()
df = clean_data(df)

# Create output folder
import os
os.makedirs("outputs", exist_ok=True)

# Set graph style
sns.set_style("whitegrid")


# 1. Diabetes outcome count
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Outcome")

plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome (0 = Negative, 1 = Positive)")
plt.ylabel("Number of Patients")

plt.savefig("outputs/outcome_distribution.png")
plt.show()


# 2. Glucose distribution
plt.figure(figsize=(7, 5))

sns.histplot(data=df, x="Glucose", kde=True)

plt.title("Glucose Distribution")
plt.xlabel("Glucose")
plt.ylabel("Frequency")

plt.savefig("outputs/glucose_distribution.png")
plt.show()


# 3. Correlation heatmap
plt.figure(figsize=(11, 8))

correlation = df.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.savefig("outputs/correlation_heatmap.png")
plt.show()


# 4. BMI compared with diabetes outcome
plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Outcome",
    y="BMI"
)

plt.title("BMI vs Diabetes Outcome")
plt.xlabel("Outcome")
plt.ylabel("BMI")

plt.savefig("outputs/bmi_outcome.png")
plt.show()


print("EDA completed successfully!")
