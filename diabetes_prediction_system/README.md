# Diabetes Prediction App

## Project Description
The Diabetes Prediction app is a web application that predicts whether a person has diabetes based on several health features such as pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age. The app utilizes a machine learning model trained on a dataset containing information about patients, including whether they have diabetes or not. 

## Data Used
The data used for training the model is stored in a CSV file named `diabetes.csv`. This dataset contains information about 768 patients, including eight features and one target variable indicating whether the patient has diabetes or not.

## Features
The features used in the dataset are as follows:
1. Pregnancies
2. Glucose
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. BMI (Body Mass Index)
7. Diabetes Pedigree Function
8. Age

## Model Used
The machine learning model used in the app is a Random Forest Classifier. This model is chosen for its ability to handle tabular data and provide accurate predictions for binary classification tasks like diabetes prediction.

## Model Evaluation
The model's performance is evaluated using accuracy score, which measures the percentage of correctly predicted outcomes out of all predictions made on the test dataset.

## Accuracy
The accuracy of the initial model is displayed in the app interface. Additionally, the accuracy of the updated model after retraining is also displayed.

## How to Run
To run the Diabetes Prediction app locally, follow these steps:
1. Clone the repository to your local machine.
2. Install the required dependencies listed in `requirements.txt`.
3. Run the app using the following command:
   ```
   streamlit run diabetes_prediction_app.py
   ```
4. Access the app through the URL displayed in the terminal after running the command.

Alternatively, you can access the deployed app on Streamlit through the following URL: [Diabetes Prediction App](https://share.streamlit.io/your-username/diabetes-prediction-app/main/app.py)

## Accessing the Saved Model
To access the trained machine learning model, use the following code snippet:

```python
import joblib

# Load the saved model
loaded_model = joblib.load('model.pkl')

# Use the loaded model for prediction
# Example:
# prediction = loaded_model.predict(user_data)
```

## Interacting with the Model
Users can interact with the model by inputting their health information through the sidebar in the app interface. After entering the required information, the app will display the prediction result along with visualizations comparing the user's data with the dataset.

## Examples and Demonstrations
Below are some screenshots demonstrating the app's UI:

![](images/1.PNG)
![](images/2.PNG)
![](images/3.PNG)
![](images/4.PNG)
