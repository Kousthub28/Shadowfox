import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from tkinter import Tk, Label, Button, filedialog, Text, Scrollbar, END
import tkinter.messagebox as messagebox
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Data Preprocessing
def preprocess_data(data):
    # Handle missing values (drop rows with NaN values)
    data = data.dropna()

    # Separate features and target variable
    X = data.drop("MEDV", axis=1)  # Target column is 'MEDV'
    y = data["MEDV"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns

# Train and evaluate the model
def train_and_evaluate(X, y, feature_names):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    accuracy = explained_variance_score(y_test, y_pred)  # Accuracy metric for regression

    # Feature importance
    feature_importance = rf.feature_importances_

    return rf, mae, mse, rmse, r2, accuracy, y_test, y_pred, feature_importance, feature_names

# GUI Implementation
class BostonHousePriceApp:
    def __init__(self, master):
        self.master = master
        master.title("Boston House Price Prediction")

        self.label = Label(master, text="Boston House Price Prediction", font=("Helvetica", 16))
        self.label.pack()

        self.load_button = Button(master, text="Load Dataset", command=self.load_dataset)
        self.load_button.pack()

        self.train_button = Button(master, text="Train Model", command=self.train_model, state="disabled")
        self.train_button.pack()

        self.export_button = Button(master, text="Export Predictions", command=self.export_predictions, state="disabled")
        self.export_button.pack()

        self.results_label = Label(master, text="Results:", font=("Helvetica", 14))
        self.results_label.pack()

        self.results_text = Text(master, height=15, width=80)
        self.results_text.pack()

        self.scrollbar = Scrollbar(master, command=self.results_text.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.results_text.configure(yscrollcommand=self.scrollbar.set)

        self.visualize_button = Button(master, text="Visualize Data", command=self.visualize_data, state="disabled")
        self.visualize_button.pack()

    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.data = load_data(file_path)
                self.results_text.insert(END, "Dataset Loaded Successfully!\n")
                self.results_text.insert(END, f"Columns: {', '.join(self.data.columns)}\n")
                self.train_button.config(state="normal")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {e}")

    def train_model(self):
        try:
            X, y, self.feature_names = preprocess_data(self.data)
            self.model, mae, mse, rmse, r2, accuracy, self.y_test, self.y_pred, self.feature_importance, feature_names = train_and_evaluate(X, y, self.feature_names)

            # Display results in the interface
            self.results_text.insert(END, "Model Trained Successfully!\n")
            self.results_text.insert(END, f"Mean Absolute Error (MAE): {mae:.2f}\n")
            self.results_text.insert(END, f"Mean Squared Error (MSE): {mse:.2f}\n")
            self.results_text.insert(END, f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
            self.results_text.insert(END, f"R² Score: {r2:.2f}\n")
            self.results_text.insert(END, f"Accuracy (Explained Variance Score): {accuracy:.2f}\n")

            self.export_button.config(state="normal")
            self.visualize_button.config(state="normal")
            messagebox.showinfo("Success", "Model training and evaluation completed!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {e}")

    def export_predictions(self):
        try:
            export_file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if export_file_path:
                predictions_df = pd.DataFrame({"Actual": self.y_test, "Predicted": self.y_pred})
                predictions_df.to_csv(export_file_path, index=False)
                messagebox.showinfo("Success", f"Predictions exported to {export_file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export predictions: {e}")

    def visualize_data(self):
        try:
            # Correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(self.data.corr(), annot=True, cmap="coolwarm")
            plt.title("Feature Correlation Heatmap")
            plt.show()

            # Actual vs Predicted scatter plot
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=self.y_test, y=self.y_pred, alpha=0.6)
            plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], color="red", lw=2)
            plt.xlabel("Actual Prices")
            plt.ylabel("Predicted Prices")
            plt.title("Actual vs Predicted Prices")
            plt.show()

            # Feature importance bar plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x=self.feature_importance, y=self.feature_names)
            plt.title("Feature Importance")
            plt.xlabel("Importance")
            plt.ylabel("Features")
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to visualize data: {e}")

# Main loop
if __name__ == "__main__":
    root = Tk()
    app = BostonHousePriceApp(root)
    root.mainloop()
  