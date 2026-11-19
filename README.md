# Diabetes Prediction Using Machine Learning

A machine learning project that analyses patient health measurements and predicts a diabetes outcome using **Logistic Regression** and **Random Forest** algorithms.

This project includes data preprocessing, exploratory data analysis, model training, model evaluation, hyperparameter tuning, and an interactive web application developed using Streamlit.

> **Disclaimer:** This project is developed for educational purposes only. Its predictions are not medical advice and must not be used to diagnose diabetes or replace evaluation by a qualified healthcare professional.

---

## Project Overview

Diabetes is a chronic health condition that affects how the body processes blood glucose. Early identification of potential diabetes cases can help people seek appropriate medical evaluation.

The objective of this project is to build a machine learning classification system that predicts whether a person has a positive or negative diabetes outcome based on health-related measurements.

The application uses two machine learning algorithms:

- Logistic Regression
- Random Forest Classifier

The trained model predicts:

- **0 – Negative diabetes outcome**
- **1 – Positive diabetes outcome**

The project uses the Pima Indians Diabetes dataset, which contains 768 records, eight input features, and one target column.

---

## Features

- Loads the diabetes dataset from a local CSV file.
- Downloads the dataset automatically if the local file is unavailable.
- Handles missing and invalid measurement values.
- Performs exploratory data analysis (EDA).
- Visualizes data distributions and feature correlations.
- Trains Logistic Regression and Random Forest models.
- Evaluates models using multiple performance metrics.
- Generates confusion matrices and feature importance visualizations.
- Performs hyperparameter tuning using GridSearchCV.
- Saves trained machine learning models.
- Provides an interactive Streamlit web application.
- Allows users to select a model and enter patient measurements to obtain a prediction.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and model evaluation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive web application |
| Joblib | Saving and loading trained models |

---

## Dataset

This project uses the **Pima Indians Diabetes dataset**.

- Total records: 768
- Input features: 8
- Target column: Outcome
- Problem type: Binary classification

### Input Features

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skinfold thickness |
| Insulin | Two-hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age in years |
| Outcome | Diabetes outcome (0 or 1) |

The dataset is included in the `data` folder.

If the local dataset is unavailable, the project attempts to download it from:

https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv

---

## Project Structure

```text
Diabetes_Prediction_ML/
│
├── app.py
├── train_model.py
├── tune_model.py
├── data_utils.py
├── eda.py
├── requirements.txt
│
├── data/
│   └── diabetes.csv
│
├── models/
│   └── diabetes_models.pkl
│
├── outputs/
│   ├── bmi_outcome.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── glucose_distribution.png
│   ├── outcome_distribution.png
│   ├── logistic_regression_confusion_matrix.png
│   ├── random_forest_confusion_matrix.png
│   ├── model_comparison.csv
│   ├── model_metrics.json
│   ├── tuned_model_comparison.csv
│   └── tuned_model_metrics.json
│
└── README.md
```

### File Descriptions

| File | Description |
|---|---|
| `app.py` | Runs the Streamlit web application |
| `train_model.py` | Preprocesses data, trains models, and evaluates performance |
| `tune_model.py` | Performs hyperparameter tuning |
| `data_utils.py` | Handles dataset loading and preprocessing |
| `eda.py` | Performs exploratory data analysis |
| `requirements.txt` | Contains the required Python libraries |
| `diabetes.csv` | Dataset used for model training |
| `diabetes_models.pkl` | Saved machine learning models |
| `outputs/` | Stores evaluation results and visualizations |

---

## Installation and Setup

Follow these steps to run the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/diabetes-prediction-using-machine-learning.git
```

Navigate into the project directory:

```bash
cd diabetes-prediction-using-machine-learning
```

Replace `YOUR-GITHUB-USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

For Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Project

### Step 1: Train the Machine Learning Models

Run:

```bash
python train_model.py
```

This script:

- Loads and preprocesses the dataset.
- Splits the data into training and testing sets.
- Trains Logistic Regression and Random Forest models.
- Evaluates their performance.
- Generates confusion matrices and feature importance visualizations.
- Saves the trained models and evaluation results.

### Step 2: Perform Exploratory Data Analysis

Run:

```bash
python eda.py
```

This generates visualizations to understand the dataset and its features.

### Step 3: Perform Hyperparameter Tuning

Run:

```bash
python tune_model.py
```

This script uses GridSearchCV and stratified cross-validation to identify suitable model hyperparameters.

### Step 4: Launch the Web Application

Run:

```bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser to access the application.

### Using the Application

1. Open the Streamlit application.
2. Select Logistic Regression or Random Forest.
3. Enter the required health measurements.
4. Click the prediction button.
5. View the predicted diabetes outcome and estimated positive-class probability.

---

## Model Development

### 1. Data Preprocessing

The project prepares the dataset before training.

- Loads the CSV dataset using Pandas.
- Treats zero values in selected measurement columns as missing values.
- Uses median imputation to handle missing values.
- Separates input features from the target column.

The selected columns are:

- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

### 2. Train-Test Split

The dataset is divided into:

- 80% training data
- 20% testing data

A stratified split is used to preserve the class distribution.

The random state is set to 42 for reproducibility.

### 3. Machine Learning Algorithms

**Logistic Regression**

Logistic Regression is a classification algorithm used to predict a binary outcome. It estimates the probability of a positive class based on input features.

The project uses median imputation, feature scaling, and Logistic Regression in a Scikit-learn pipeline.

**Random Forest**

Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions.

The project uses 200 decision trees and balanced class weights in its initial model configuration.

### 4. Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

### 5. Hyperparameter Tuning

The project uses:

- GridSearchCV
- Stratified K-Fold Cross-Validation
- Five cross-validation folds
- F1-score as the tuning metric

This helps compare different model configurations.

---

## Model Performance

The following results are taken from the evaluation files generated by the project.

### Initial Model Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 70.78% | 60.00% | 50.00% | 54.55% |
| Random Forest | 74.03% | 61.67% | 68.52% | 64.91% |

### Results After Hyperparameter Tuning

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 70.78% | 57.14% | 66.67% | 61.54% |
| Random Forest | 72.08% | 62.79% | 50.00% | 55.67% |

### Observation

In the initial evaluation, Random Forest achieved higher accuracy, recall, and F1-score than Logistic Regression.

After hyperparameter tuning, Logistic Regression achieved a higher recall and F1-score than its initial version, while Random Forest's test-set performance decreased on some metrics.

These results show why model performance should be evaluated using multiple metrics rather than accuracy alone.

---

## Exploratory Data Analysis and Visualizations

The project generates several visualizations to understand the dataset and model behaviour.

### 1. Outcome Distribution

Shows the distribution of positive and negative diabetes outcomes.

![Outcome Distribution](outputs/outcome_distribution.png)

### 2. Glucose Distribution

Shows the distribution of glucose measurements in the dataset.

![Glucose Distribution](outputs/glucose_distribution.png)

### 3. Correlation Heatmap

Displays correlations between the dataset features.

![Correlation Heatmap](outputs/correlation_heatmap.png)

### 4. BMI and Outcome

Visualizes the relationship between BMI and diabetes outcome.

![BMI and Outcome](outputs/bmi_outcome.png)

### 5. Feature Importance

Displays the feature importance generated by the Random Forest model.

![Feature Importance](outputs/feature_importance.png)

### 6. Logistic Regression Confusion Matrix

![Logistic Regression Confusion Matrix](outputs/logistic_regression_confusion_matrix.png)

### 7. Random Forest Confusion Matrix

![Random Forest Confusion Matrix](outputs/random_forest_confusion_matrix.png)

---

## Important Notes

- Run `train_model.py` before launching the Streamlit application if the saved model files are unavailable.
- The application uses the saved model file in the `models` folder.
- The application reads evaluation metrics from `outputs/model_metrics.json`.
- `tune_model.py` saves tuned models to the same model file and writes separate tuned metrics to `outputs/tuned_model_metrics.json`.
- The current application reads the original metrics file, so after tuning, the displayed metrics may not correspond to the tuned models.
- The probability displayed by the application is a model-estimated positive-class probability. It is not a clinically validated risk score.
- Model performance depends on the dataset, preprocessing, training configuration, and test split.

---

## Future Improvements

- Use larger and more diverse datasets.
- Compare additional machine learning algorithms.
- Improve model validation and probability calibration.
- Add explainable AI techniques to understand predictions.
- Improve the user interface and application design.
- Deploy the application online.
- Add automated testing and model version tracking.

---

## Author

**Poojitha V**

B.Tech – Computer Science and Engineering

---

## Acknowledgements

- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit
- Plotly Datasets

---

**If you find this project useful, consider giving the repository a star!**