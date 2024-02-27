# Diabetes Prediction System

## Overview
The **Diabetes Prediction System** is a machine learning project aimed at predicting whether an individual has diabetes based on various health features. The project utilizes the PIMA Diabetes Dataset, which contains records of patients along with relevant features.

## Dataset
- **Dataset Name**: PIMA Diabetes Dataset
- **Source**: [Download the dataset here](https://www.dropbox.com/scl/fi/kwj13e60aeiigsddk7aap/diabetes.csv?rlkey=ptbmxfp7ezp2hncuzihux9kb5&dl=0)
- **Features**: Includes 8 health-related features (e.g., glucose level, blood pressure, BMI) and 1 outcome label (diabetes or not).

## Key Findings
1. **Exploratory Data Analysis (EDA)**:
   - Explored data distribution, missing values, and correlations.
   - Visualized feature relationships.

2. **Data Preprocessing**:
   - Standardized features using the `StandardScaler`.
   - Split data into training and test sets.

3. **Model Training**:
   - Used a Support Vector Machine (SVM) classifier with a linear kernel.
   - Achieved accuracy on both training and test sets.

4. **Predictive System**:
   - Demonstrated how to make predictions using the trained model.
   - Saved the model using the `pickle` module.

## Instructions
1. **Run the Notebook**:
   - Open the Jupyter notebook provided.
   - Execute each cell sequentially to load the dataset, preprocess data, train the model, and save it.

2. **Load the Saved Model**:
   - Use the following code snippet to load the saved model:
     ```python
     with open('classifier.pkl', 'rb') as file:
         loaded_model = pickle.load(file)
     ```

Feel free to explore the notebook and adapt it for your own use! 📊🤖
