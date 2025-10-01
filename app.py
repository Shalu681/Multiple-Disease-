import streamlit as st
import pandas as pd
import pickle
import os
import json

# --- Paths for model files, update accordingly ---
BASE_DIR = "/content/drive/MyDrive/2078_Akalya_Multiple Disease"

MODEL_PATHS = {
    "Parkinson's Disease": os.path.join(BASE_DIR, "parkinson_models"),
    "Kidney Disease": os.path.join(BASE_DIR, "kidney_models"),
    "Liver Disease": os.path.join(BASE_DIR, "liver_models"),
    "Diabetes": os.path.join(BASE_DIR, "diabetes_models"),
    "Heart Disease": os.path.join(BASE_DIR, "heart_models"),
}

# --- Persistent user store file ---
USER_FILE = "users.json"

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

users = load_users()

def register(username, password):
    if username in users:
        return False, "User already exists."
    users[username] = password
    save_users(users)
    return True, "Registration successful."

def login(username, password):
    if username in users and users[username] == password:
        return True, "Login successful."
    return False, "Invalid username or password."

# --- Load model, scaler, PCA for Parkinson's as example ---
@st.cache_data(show_spinner=False)
def load_parkinson_artifacts():
    model_dir = MODEL_PATHS["Parkinson's Disease"]
    model = pickle.load(open(os.path.join(model_dir, "XGBoost_model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(model_dir, "scaler.pkl"), "rb"))
    pca = pickle.load(open(os.path.join(model_dir, "pca.pkl"), "rb"))
    return model, scaler, pca

def predict_parkinsons(input_dict):
    model, scaler, pca = load_parkinson_artifacts()
    features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
        "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3",
        "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2",
        "D2", "PPE"
    ]
    df = pd.DataFrame([input_dict])
    for f in features:
        if f not in df.columns:
            df[f] = 0
    df = df[features]
    X_scaled = scaler.transform(df)
    X_pca = pca.transform(X_scaled)
    pred = model.predict(X_pca)[0]
    proba = model.predict_proba(X_pca)[0][1]
    return pred, proba

# Placeholder prediction functions for other diseases
def predict_kidney(input_dict):
    return 0, 0.1

def predict_liver(input_dict):
    return 0, 0.2

def predict_diabetes(input_dict):
    return 1, 0.8

def predict_heart(input_dict):
    return 0, 0.3

DISEASES = {
    "Parkinson's Disease": {
        "features": [
            "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
            "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3",
            "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2",
            "D2", "PPE"
        ],
        "predict_fn": predict_parkinsons
    },
    "Kidney Disease": {
        "features": [
            "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu",
            "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"
        ],
        "predict_fn": predict_kidney
    },
    "Liver Disease": {
        "features": [
            "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase",
            "Alamine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Protiens",
            "Albumin", "Albumin_and_Globulin_Ratio"
        ],
        "predict_fn": predict_liver
    },
    "Diabetes": {
        "features": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigreeFunction", "Age"
        ],
        "predict_fn": predict_diabetes
    },
    "Heart Disease": {
        "features": [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak",
            "slope", "ca", "thal"
        ],
        "predict_fn": predict_heart
    }
}

# --- Streamlit UI ---
st.title("Multi-Disease Risk Prediction App")

page = st.sidebar.radio("Navigate", ["Home", "Login/Register", "Prediction"])

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

if page == "Home":
    st.header("Welcome to the Health Risk Prediction App")
    st.write("""
        This app predicts your individual risk for:
        - Parkinson's Disease
        - Kidney Disease
        - Liver Disease
        - Diabetes
        - Heart Disease
    """)
    if st.session_state.logged_in:
        st.success(f"Logged in as {st.session_state.username}")
    else:
        st.info("Please login or register to use the prediction features.")

elif page == "Login/Register":
    st.header("Login or Register")

    option = st.radio("Choose action:", ["Login", "Register"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button(option):
        if username == "" or password == "":
            st.error("Username and password cannot be empty.")
        else:
            if option == "Register":
                success, msg = register(username, password)
                st.write(f"Register attempt: username={username}, success={success}, msg={msg}")
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
            else:
                success, msg = login(username, password)
                st.write(f"Login attempt: username={username}, success={success}, msg={msg}")
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(msg)
                else:
                    st.error(msg)

elif page == "Prediction":
    if not st.session_state.logged_in:
        st.warning("You must be logged in to use the prediction page.")
        st.stop()

    st.header("Disease Risk Prediction")

    disease = st.selectbox("Select disease to predict:", list(DISEASES.keys()))

    features = DISEASES[disease]["features"]
    predict_fn = DISEASES[disease]["predict_fn"]

    st.subheader(f"Enter {disease} features")

    user_input = {}
    all_filled = True

    for feature in features:
        # For Gender or sex fields, you might want to use selectbox or text input, but here using number_input for simplicity
        val = st.number_input(feature, format="%.6f", step=0.01)
        user_input[feature] = val
        if val == 0 or val == 0.0:
            all_filled = False

    if st.button(f"Predict {disease} Risk"):
        if not all_filled:
            st.error("Please fill in all feature fields with non-zero values.")
        else:
            pred, conf = predict_fn(user_input)
            if pred == 1:
                st.error(f"Prediction: {disease} Positive (Risk Detected)\nConfidence: {conf:.2f}")
            else:
                st.success(f"Prediction: {disease} Negative (No Risk Detected)\nConfidence: {conf:.2f}")

# Show registered users for debug (optional, remove in prod)
st.sidebar.write("Registered users:", list(users.keys()))
