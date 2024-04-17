# Import the required libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # for saving and loading models
import time

# Function to load data and split it into features and target variable
def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1]  # Features (all columns except the last one)
    y = df.iloc[:, -1]   # Target variable (last column)
    return df, X, y

# Function to standardize input data
def standardize_input_data(user_data, scaler):
    standardized_data = scaler.transform(user_data)
    return standardized_data

# Function to train the model
def train_model(X_train, y_train):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    return test_accuracy

# Function to check if it's time to retrain the model
def is_time_to_retrain(retraining_trigger):
    if retraining_trigger == "weekly":
        current_day_of_week = time.strftime("%A", time.localtime())
        if current_day_of_week == "Sunday":
            return True
    return False

# Function to introduce a delay before checking the trigger again
def wait_for_next_trigger():
    time.sleep(86400)  # Wait for 24 hours (86400 seconds)

# Function to display user report
def display_user_report(user_data, user_result, df, model, X_test, y_test):
    st.title('Your Report:')
    output = 'You are Not Diabetic' if user_result == 0 else 'You are Diabetic'
    st.header(output)
    st.subheader('Accuracy:')
    st.write(f"{evaluate_model(model, X_test, y_test)*100:.2f}%")

    # Visualisations
    st.header('________________________________________')
    st.title('Visualised Patient Report')
    st.subheader('Blue dot = Not Diabetic')
    st.subheader('Red dot = Diabetic')

    if user_result == 0:
        color = 'blue'
    else:
        color = 'red'

    # Plot Age vs Pregnancies
    st.subheader('Pregnancy count Graph (Others vs Yours)')
    fig_preg = plt.figure()
    ax1 = sns.scatterplot(x='Age', y='Pregnancies', data=df, hue='Outcome', palette='Greens')
    ax2 = sns.scatterplot(x=user_data['Age'], y=user_data['Pregnancies'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 20, 2))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_preg)

    # Plot Age vs Glucose
    st.subheader('Glucose Value Graph (Others vs Yours)')
    fig_glucose = plt.figure()
    ax3 = sns.scatterplot(x='Age', y='Glucose', data=df, hue='Outcome', palette='rocket')
    ax4 = sns.scatterplot(x=user_data['Age'], y=user_data['Glucose'], s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 220, 10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_glucose)

    # Additional plots...

# Main app code
def main():
    st.title('Diabetes Checkup')

    # Load the Data
    file_path = 'data/diabetes.csv'
    df, X, y = load_data(file_path)
    st.sidebar.header('Patient Data')
    st.subheader('Training Data Stats')
    st.write(X.describe())

    # Standardize the data
    scaler = StandardScaler()
    scaler.fit(X)

    # Sidebar function to get user input
    def user_report():
        pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
        glucose = st.sidebar.slider('Glucose', 0, 200, 120)
        bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
        skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
        insulin = st.sidebar.slider('Insulin', 0, 846, 79)
        bmi = st.sidebar.slider('BMI', 0, 67, 20)
        dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
        age = st.sidebar.slider('Age', 21, 88, 33)

        user_report_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': bp,
            'SkinThickness': skinthickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        report_data = pd.DataFrame(user_report_data, index=[0])
        return report_data

    # Patient Data
    user_data = user_report()
    st.subheader('Patient Data')
    st.write(user_data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Initial Model
    initial_model = train_model(X_train, y_train)
    initial_accuracy = evaluate_model(initial_model, X_test, y_test)
    st.write("Initial Model Accuracy:", initial_accuracy)

    # Define the Trigger for Retraining
    retraining_trigger = "weekly"

    # Retraining Loop
    while True:
        # Check if it's time to retrain the model
        if is_time_to_retrain(retraining_trigger):
            # Retrain the Model
            updated_model = train_model(X, y)
            updated_accuracy = evaluate_model(updated_model, X_test, y_test)
            st.write("Updated Model Accuracy:", updated_accuracy)

            # Save the Updated Model
            joblib.dump(updated_model, 'model.pkl')

            # Model Prediction
            user_data_standardized = standardize_input_data(user_data, scaler)
            user_result = updated_model.predict(user_data_standardized)

            # Display User Report
            display_user_report(user_data, user_result, df, updated_model, X_test, y_test)

        # Optionally, add a delay before checking the trigger again
        wait_for_next_trigger()


# Run the main app
if __name__ == "__main__":
    main()
