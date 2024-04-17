# Data Directory

## Overview
The data directory contains the dataset used for training and testing the Diabetes Prediction app.

## Data Description
The dataset contains information about patients, including various health features and a target variable indicating whether the patient has diabetes or not.

The target variable is binary, with values indicating:
- 0: No diabetes
- 1: Diabetes

## Data Splitting
In the main code (`diabetes_prediction_app.py`), the dataset is split into training and testing sets using the `train_test_split` function from scikit-learn. The splitting ratio is set to 80% training data and 20% testing data (`test_size=0.2`). This ensures that a portion of the data is reserved for evaluating the model's performance.

```python
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

The training set (`X_train` and `y_train`) is used to train the machine learning model, while the testing set (`X_test` and `y_test`) is used to evaluate the model's accuracy and performance.
