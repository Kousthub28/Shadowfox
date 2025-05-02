# Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from scipy.stats import zscore
from flask import Flask, render_template, request
import io
import base64

# Data Preprocessing
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    print("Dataset Loaded Successfully!")

    # Handling Missing Values
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)

    # Handling Outliers using Z-Score
    numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df = df[(np.abs(zscore(df[numerical_cols])) < 3).all(axis=1)]

    # One-Hot Encoding Categorical Variables
    encoder = OneHotEncoder(sparse=False, drop="first")
    categorical_cols = df.select_dtypes(include=["object"]).columns
    categorical_cols = [col for col in categorical_cols if col != "Loan_Status"]
    encoded = pd.DataFrame(encoder.fit_transform(df[categorical_cols]),
                           columns=encoder.get_feature_names_out(categorical_cols))
    df = pd.concat([df.drop(categorical_cols, axis=1), encoded], axis=1)

    # Map Target Column
    if "Loan_Status" in df.columns:
        df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    # Feature Engineering
    df["Debt_Income_Ratio"] = df["LoanAmount"] / (df["ApplicantIncome"] + 1)
    df["Loan_to_Income_Ratio"] = df["LoanAmount"] / (df["ApplicantIncome"] + df["CoapplicantIncome"] + 1)

    # Scaling Numerical Features
    scaler = StandardScaler()
    numerical_features = df.select_dtypes(include=["float64", "int64"]).columns
    numerical_features = [col for col in numerical_features if col != "Loan_Status"]
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df

# Exploratory Data Analysis (EDA)
def perform_eda(df):
    print("\nPerforming EDA...")

    # Loan Status Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Loan_Status", data=df, palette="pastel")
    plt.title("Loan Status Distribution")
    plt.savefig("static/loan_status_distribution.png")
    plt.close()

    # Applicant Income Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["ApplicantIncome"], kde=True, bins=30, color="blue")
    plt.title("Applicant Income Distribution")
    plt.xlabel("Applicant Income")
    plt.ylabel("Frequency")
    plt.savefig("static/applicant_income_distribution.png")
    plt.close()

    # Save EDA Report
    with open("eda_report.txt", "w") as file:
        file.write("### EDA Report ###\n\n")
        file.write("1. Loan Status Distribution saved as 'static/loan_status_distribution.png'.\n")
        file.write("2. Applicant Income Distribution saved as 'static/applicant_income_distribution.png'.\n")
    print("EDA Report Generated.")

# Model Training
def train_ai_models(data_path):
    df = load_and_preprocess_data(data_path)
    perform_eda(df)

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train RandomForestClassifier
    print("Training RandomForestClassifier...")
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    print(f"RandomForest Accuracy: {rf_accuracy}")

    # Train SVC
    print("Training SVC...")
    svc_model = SVC(probability=True, random_state=42)
    svc_model.fit(X_train, y_train)
    svc_accuracy = accuracy_score(y_test, svc_model.predict(X_test))
    print(f"SVC Accuracy: {svc_accuracy}")

    # Train XGBoost
    print("Training XGBoost...")
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
    xgb_model.fit(X_train, y_train)
    xgb_accuracy = accuracy_score(y_test, xgb_model.predict(X_test))
    print(f"XGBoost Accuracy: {xgb_accuracy}")

    # Select Best Model
    best_model = max([(rf_model, rf_accuracy), (svc_model, svc_accuracy), (xgb_model, xgb_accuracy)], key=lambda x: x[1])[0]
    joblib.dump(best_model, "best_loan_model.pkl")
    print("Best Model Saved as 'best_loan_model.pkl'.")

    # Interpret Model with SHAP
    explainer = shap.TreeExplainer(best_model)
    joblib.dump(explainer, "shap_explainer.pkl")
    print("SHAP Explainer Saved.")

    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("static/shap_summary_plot.png")
    plt.close()

# Flask Application
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    model = joblib.load("best_loan_model.pkl")
    explainer = joblib.load("shap_explainer.pkl")

    input_data = {
        "Gender": int(request.form.get("Gender", 0)),
        "Married": int(request.form.get("Married", 0)),
        "Dependents": int(request.form.get("Dependents", 0)),
        "Education": int(request.form.get("Education", 0)),
        "Self_Employed": int(request.form.get("Self_Employed", 0)),
        "ApplicantIncome": float(request.form.get("ApplicantIncome", 0)),
        "CoapplicantIncome": float(request.form.get("CoapplicantIncome", 0)),
        "LoanAmount": float(request.form.get("LoanAmount", 0)),
        "Loan_Amount_Term": float(request.form.get("Loan_Amount_Term", 360)),
        "Credit_History": float(request.form.get("Credit_History", 1)),
        "Property_Area": int(request.form.get("Property_Area", 0))
    }

    input_data["Debt_Income_Ratio"] = input_data["LoanAmount"] / (input_data["ApplicantIncome"] + input_data["CoapplicantIncome"] + 1)
    input_data["Loan_to_Income_Ratio"] = input_data["LoanAmount"] / (input_data["ApplicantIncome"] + input_data["CoapplicantIncome"] + 1)

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    result = "Loan Approved" if prediction[0] == 1 else "Loan Rejected"

    shap_values = explainer.shap_values(input_df)[0]
    top_reasons = sorted(zip(input_df.columns, shap_values), key=lambda x: abs(x[1]), reverse=True)[:3]

    return render_template("index.html", prediction_text=f"Prediction: {result}", reasons=top_reasons)

if __name__ == "__main__":
    train_ai_models("loan_prediction.csv")
    app.run(debug=True)