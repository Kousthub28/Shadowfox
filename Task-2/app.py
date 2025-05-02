# Import necessary libraries
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Matplotlib
from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and SHAP explainer
model = joblib.load("best_loan_model.pkl")
explainer = joblib.load("shap_explainer.pkl")

# Mock function to calculate CIBIL score
def calculate_cibil_score(input_data):
    base_score = 300  # Minimum CIBIL score
    max_score = 900   # Maximum CIBIL score
    debt_factor = max(0, 1 - input_data["Debt_Income_Ratio"])
    income_factor = min(input_data["ApplicantIncome"] / 10000, 1)
    cibil_score = int(base_score + (max_score - base_score) * (0.5 * income_factor + 0.5 * debt_factor))
    return cibil_score

# Helper function to create a pie chart for Credit History Distribution
def create_credit_history_chart(credit_history_stats):
    labels = ['Good Credit', 'Bad Credit']
    sizes = [credit_history_stats['good_credit'], credit_history_stats['bad_credit']]
    colors = ['#28a745', '#dc3545']
    explode = (0.1, 0)  # Explode the first slice (Good Credit)
    plt.figure(figsize=(5, 5))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Credit History Distribution')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return chart_base64

# Helper function to create a histogram for CIBIL Score Distribution
def create_cibil_score_chart(cibil_scores):
    plt.figure(figsize=(8, 5))
    sns.histplot(cibil_scores, kde=True, bins=20, color='blue')
    plt.title('CIBIL Score Distribution')
    plt.xlabel('CIBIL Score')
    plt.ylabel('Frequency')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return chart_base64

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input from the form
        input_data = {
            "Gender": int(request.form.get('Gender', 0)),
            "Married": int(request.form.get('Married', 0)),
            "Dependents": int(request.form.get('Dependents', 0)),
            "Education": int(request.form.get('Education', 0)),
            "Self_Employed": int(request.form.get('Self_Employed', 0)),
            "ApplicantIncome": float(request.form.get('ApplicantIncome', 0)),
            "CoapplicantIncome": float(request.form.get('CoapplicantIncome', 0)),
            "LoanAmount": float(request.form.get('LoanAmount', 0)),
            "Loan_Amount_Term": float(request.form.get('Loan_Amount_Term', 360)),
            "Credit_History": float(request.form.get('Credit_History', 1)),
            "Property_Area": int(request.form.get('Property_Area', 0))
        }

        # Add calculated features
        input_data["Debt_Income_Ratio"] = input_data["LoanAmount"] / (input_data["ApplicantIncome"] + input_data["CoapplicantIncome"] + 1)
        input_data["Loan_to_Income_Ratio"] = input_data["LoanAmount"] / (input_data["ApplicantIncome"] + input_data["CoapplicantIncome"] + 1)

        # Convert input data into a DataFrame
        input_df = pd.DataFrame([input_data])

        # Predict loan approval
        prediction = model.predict(input_df)

        # Determine approval/rejection result
        result = "Loan Approved" if prediction[0] == 1 else "Loan Rejected"

        # Credit History Insights
        credit_history = input_data["Credit_History"]
        credit_history_insight = "Good" if credit_history == 1 else "Bad"

        # Calculate CIBIL score
        cibil_score = calculate_cibil_score(input_data)

        # Get SHAP values for explanation
        shap_values = explainer.shap_values(input_df)[0]
        top_reasons = sorted(zip(input_df.columns, shap_values), key=lambda x: abs(x[1]), reverse=True)[:3]

        # Mock Credit History Stats for visualization
        credit_history_stats = {"good_credit": 70, "bad_credit": 30}  # Example data
        credit_history_chart = create_credit_history_chart(credit_history_stats)

        # Mock CIBIL Scores for visualization
        mock_cibil_scores = [calculate_cibil_score({"ApplicantIncome": i, "Debt_Income_Ratio": 0.2}) for i in range(10000, 100000, 2000)]
        cibil_score_chart = create_cibil_score_chart(mock_cibil_scores)

        # Render the template with results and visualizations
        return render_template(
            'index.html',
            prediction_text=f"Prediction: {result}",
            cibil_score=f"CIBIL Score: {cibil_score}",
            credit_history_insight=f"Credit History: {credit_history_insight}",
            reasons=top_reasons,
            credit_history_chart=credit_history_chart,
            cibil_score_chart=cibil_score_chart
        )

    except Exception as e:
        return f"An error occurred: {e}", 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)