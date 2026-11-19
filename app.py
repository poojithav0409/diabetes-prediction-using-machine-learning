
import streamlit as st
import pandas as pd
import joblib
import json
import os


# ---------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)


# ---------------------------------------
# LOAD TRAINED MODELS
# ---------------------------------------

@st.cache_resource
def load_models():

    return joblib.load("models/diabetes_models.pkl")


if not os.path.exists("models/diabetes_models.pkl"):

    st.error(
        "Trained models not found. "
        "Please run train_model.py first."
    )

    st.stop()


models = load_models()


# ---------------------------------------
# LOAD EVALUATION RESULTS
# ---------------------------------------

metrics_path = "outputs/model_metrics.json"

if os.path.exists(metrics_path):

    with open(metrics_path, "r") as file:

        metrics = json.load(file)

else:

    metrics = {}


# ---------------------------------------
# APPLICATION TITLE
# ---------------------------------------

st.title("🩺 Diabetes Prediction System")

st.write(
    "Enter the patient health measurements "
    "to get an educational machine learning prediction."
)

st.warning(
    "This application is for educational purposes only. "
    "It cannot diagnose diabetes or replace medical advice."
)


# ---------------------------------------
# MODEL SELECTION
# ---------------------------------------

st.subheader("Select Machine Learning Model")

selected_model = st.selectbox(
    "Choose a model",
    [
        "Logistic Regression",
        "Random Forest"
    ]
)


# ---------------------------------------
# PATIENT INPUT SECTION
# ---------------------------------------

st.subheader("Enter Patient Measurements")

st.info(
    "You can type values directly into the fields "
    "or use the + and - buttons."
)

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=2,
        step=1,
        format="%d"
    )

    glucose = st.number_input(
        "Glucose",
        min_value=1.0,
        max_value=300.0,
        value=120.0,
        step=0.1,
        format="%.2f"
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=1.0,
        max_value=200.0,
        value=70.0,
        step=0.1,
        format="%.2f"
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=1.0,
        max_value=100.0,
        value=25.0,
        step=0.1,
        format="%.2f"
    )


with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=1.0,
        max_value=1000.0,
        value=80.0,
        step=0.1,
        format="%.2f"
    )

    bmi = st.number_input(
        "BMI",
        min_value=1.0,
        max_value=100.0,
        value=28.5,
        step=0.1,
        format="%.2f"
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.01,
        max_value=3.0,
        value=0.5,
        step=0.01,
        format="%.2f"
    )

    age = st.number_input(
        "Age",
        min_value=21,
        max_value=100,
        value=30,
        step=1,
        format="%d"
    )

# ---------------------------------------
# PREDICTION
# ---------------------------------------

if st.button("Predict Diabetes Status", type="primary"):

    # Prepare input data
    input_data = pd.DataFrame({

        "Pregnancies": [pregnancies],

        "Glucose": [glucose],

        "BloodPressure": [blood_pressure],

        "SkinThickness": [skin_thickness],

        "Insulin": [insulin],

        "BMI": [bmi],

        "DiabetesPedigreeFunction": [
            diabetes_pedigree
        ],

        "Age": [age]

    })

    # Select the trained model
    model = models[selected_model]

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get positive-class probability
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            "The model predicts a positive diabetes outcome."
        )

    else:

        st.success(
            "The model predicts a negative diabetes outcome."
        )

    st.metric(
        "Model-estimated probability of positive outcome",
        f"{probability * 100:.2f}%"
    )

    st.caption(
        "This probability is a model estimate, not a clinical "
        "risk score or a confirmed diagnosis."
    )


# ---------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------

st.divider()

st.subheader("Model Performance")

if selected_model in metrics:

    selected_metrics = metrics[selected_model]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{selected_metrics['Accuracy'] * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{selected_metrics['Precision'] * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{selected_metrics['Recall'] * 100:.2f}%"
    )

    col4.metric(
        "F1-score",
        f"{selected_metrics['F1-score'] * 100:.2f}%"
    )

else:

    st.info(
        "Train the models to display performance metrics."
    )


# ---------------------------------------
# FOOTER
# ---------------------------------------

st.divider()

st.caption(
    "Diabetes Prediction Using Machine Learning | "
    "Python, Pandas, NumPy, Scikit-learn and Streamlit"
)
