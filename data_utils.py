
import os
import pandas as pd

# Dataset location
DATA_PATH = "data/diabetes.csv"

# Public dataset source
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "plotly/datasets/master/diabetes.csv"
)


def load_data():

    # Create data folder if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Download dataset if it is not available
    if not os.path.exists(DATA_PATH):

        print("Downloading dataset...")

        df = pd.read_csv(DATA_URL)

        df.to_csv(DATA_PATH, index=False)

        print("Dataset downloaded successfully!")

    else:

        print("Loading existing dataset...")

        df = pd.read_csv(DATA_PATH)

    return df


def clean_data(df):

    # Create a copy to protect original data
    df = df.copy()

    # These columns should not normally have zero measurements
    zero_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    # Replace invalid zeros with missing values
    df[zero_columns] = df[zero_columns].replace(0, float("nan"))

    return df


if __name__ == "__main__":

    df = load_data()

    print("\nFirst five records:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nDataset information:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())
