---

# Diabetes Prediction System

## Overview

This project aims to develop a Diabetes Prediction System using machine learning techniques. The system predicts the likelihood of an individual having diabetes based on various health parameters such as glucose level, blood pressure, BMI, etc. The objective is to aid in the early detection and management of diabetes, thereby improving health outcomes.

## Dataset

The project utilizes the PIMA Indians Diabetes Dataset, obtained from Kaggle. This dataset contains health parameters of PIMA Indian women, including pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age.

## Key Findings

After training and evaluating the predictive models, the following key findings were observed:

- The Support Vector Machine (SVM) model achieved an accuracy of 77.27% on the test data.
- The Gradient Boosting Classifier achieved an accuracy of 96.42% on the test data.
- Both models show promising results in predicting diabetes based on health parameters.

## Running the Notebook

To run the notebook:

1. Clone this repository to your local machine.
2. Ensure you have Python and Jupyter Notebook installed.
3. Navigate to the directory containing the notebook file.
4. Run `jupyter notebook` in your terminal to start the Jupyter Notebook server.
5. Open the notebook file (`Diabetes_Prediction.ipynb`) and execute the cells sequentially.

## Loading Saved Models

To load the saved models:

1. Clone or download this repository to your local machine.
2. Navigate to the directory containing the saved model files (`simple_model.pkl` and `optimized_model.pkl`).
3. Use the appropriate Python code to load the models using the `pickle` module.
   
  ```
# Load the saved simple model
with open(simple_model, 'rb') as f:
    model = pickle.load(f)


# Load the saved optimized model
with open(optimized_model, 'rb') as f:
    model = pickle.load(f)
```

## Optimization Techniques

### 1. Hyperparameter Tuning
- The `RandomizedSearchCV` technique was employed to search for the optimal hyperparameters of the models.
- Parameters such as learning rate, max depth, subsample ratio, number of estimators, gamma, alpha, and lambda were tuned.

### 2. Feature Engineering
- Feature engineering techniques were applied to create additional informative features, such as interaction terms and polynomial features.
- The `StandardScaler` was used to standardize the input features to ensure uniform scaling.

### 3. Ensemble Methods
- Ensemble methods, such as Gradient Boosting, were used to combine multiple weak learners to improve predictive performance.
- Gradient Boosting Classifier from the `xgboost` library was employed with hyperparameters tuned for optimal performance.

## Parameter Selection and Tuning

- Parameters were selected based on domain knowledge and experimentation.
- Randomized search with cross-validation was used to search for the optimal parameter values within predefined ranges.
- Parameter values were selected based on the highest cross-validation accuracy and model performance on the test data.

## Error Analysis

Error analysis was conducted using various techniques:

1. Specificity Output: Analyzing the specificity of the models to identify their ability to correctly identify non-diabetic individuals.
2. Confusion Matrix: Visualizing the confusion matrix to understand the distribution of true positive, false positive, true negative, and false negative predictions.
3. F1 Score: Calculating the F1 score to evaluate the model's precision and recall performance.

The outputs from these analyses are well-formatted and easy to interpret, providing insights into the model's strengths and weaknesses.

---
