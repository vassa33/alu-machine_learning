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
The model's performance is evaluated using an accuracy score, which measures the percentage of correctly predicted outcomes out of all predictions made on the test dataset.

## Accuracy
The accuracy of the initial model is displayed in the app interface. Additionally, the accuracy of the updated model after retraining is also displayed.

Sure, here's the updated section on how to run the Diabetes Prediction app in the main README.md:

---

## How to Run

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/vassa33/alu-machine_learning.git
   ```

2. **Navigate to the System's Directory**:
   ```bash
   cd diabetes_prediction_system
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**:
   - **Locally**: 
     ```bash
     streamlit run src/diabetes_prediction_app.py
     ```
     The `diabetes_prediction_app.py` file is located in the `src` directory.
     Access the app through the URL displayed in the terminal after running the command.
   
   - **Via Streamlit Sharing**: 
Alternatively, you can access the deployed app on Streamlit through the following URL: [Diabetes Detective App]([https://share.streamlit.io/your-username/diabetes-prediction-app/main/app.py](https://diabetes-detective.streamlit.app/))


5. **Interact with the App**:
   - Once the app is running, open the provided URL in your web browser.
   - Use the sidebar sliders to input patient data.
   - The app will display predictions for diabetes and provide visualizations of the patient's data.

## Accessing the Saved Model
To access the trained machine-learning model, use the following code snippet:

```python
import joblib

# Load the saved model
loaded_model = joblib.load('model.pkl')

# Use the loaded model for prediction
# Example:
# prediction = loaded_model.predict(user_data)
```


## Examples and Demonstrations
Below are some screenshots demonstrating the app's UI:

![](images/1.PNG)
![](images/2.PNG)
![](images/3.PNG)
![](images/4.PNG)
