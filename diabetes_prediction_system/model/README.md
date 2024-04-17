# Model Directory

## Overview
The model directory contains the trained machine learning model used for predicting diabetes in the Diabetes Prediction app.

## Model Description
The trained model achieved a testing accuracy of approximately 72.73%. It was trained using a random state of 42 for reproducibility.

## Accessing the Model
To access the saved model in Python, you can use the following code snippet:

```python
import joblib

# Load the saved model
loaded_model = joblib.load('diabetes_model.pkl')
```

## Using the Model
Once the model is loaded, you can use it to make predictions on new data. Here's an example of how to use the model to predict diabetes for a new set of patient data:

```python
# Example user data
user_data = [[3, 120, 70, 30, 90, 25, 0.47, 40]]  # Replace with actual user data

# Make predictions
predicted_outcome = loaded_model.predict(user_data)
print("Predicted outcome:", predicted_outcome)
```

This code snippet demonstrates how to load the saved model and use it to make predictions on new data. Replace the `user_data` variable with actual patient data to get predictions for real-world scenarios.
