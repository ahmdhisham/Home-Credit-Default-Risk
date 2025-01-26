# Home Credit Default Risk
In this project, we aim to develop a machine learning model to predict the likelihood of a customer defaulting on a home credit loan. leveraging past application data to help financial institutions evaluate loan eligibility and mitigate credit risk. This is a classification problem where the target variable is binary, indicating whether the customer will default (1) or not (0).


## Table of Contents
- [Overview](#overview)
- [Objective](#objective)
- [Workflow](#workflow)
- [Project Structure](#project-structure)
- [Machine Learning Techniques](#machine-learning-techniques)
- [Evaluation Results](#evaluation-results)


## Overview
Many individuals struggle to obtain loans due to insufficient or non-existent credit histories. **Home Credit** aims to enhance financial inclusion for the unbanked population by offering a secure and positive borrowing experience. This project leverages alternative data to assess clients' repayment abilities and predict which applicants are most likely to default.

### The repository contains the following components:
**Notebooks:** Comprehensive notebooks showcasing data preprocessing, exploratory data analysis (EDA) techniques, model training, and performance evaluation.

**Source Code:** Scripts dedicated to executing all data preprocessing steps, resulting in a fully cleaned dataset ready for analysis and modeling.

**Streamlit Web Application:** An interactive web application that facilitates training a CatBoost model, provides evaluation metrics, and outputs the final trained model in a (.cbm) format.


## Objective
Analyze loan applicant data provided by **Home Credit** and identify applicants most likely to default using machine learning techniques.


## Workflow
1. **Data Preprocessing**:
   - Automated using a separate Python file (`preprocessing_functions.py`) for reusable functions.
2. **Model Training**:
   - Trained multiple machine learning models to evaluate default risk.
3. **Model Evaluation**:
   - Assessed model performance using metrics such as accuracy, precision, recall, F1-score, and AUC.
4. **Model Tracking**:
   - Tracked experiments and results using **MLflow** for reproducibility.
5. **Comparison of Model Metrics**:
   - Selected the best model based on evaluation results.


## Project Structure
**Under src directory**
1. **`utils.py`**:
   - Includes reusable functions.
2. **`preprocessing.py`**:
   - Includes all the data preprocessing made in one script.
3. **`main.py`**:
   - Includes the main code of the streamlit web application to train the model.
4. **`run_scripts.bat`**:
   - Runs all the scripts once you launch it.
     
**Under notebooks directory**
5. **`home_credit_notebook_part1.ipynb`**:
   - The first part of the primary notebook, including EDA and some preprocessing.
6. **`home_credit_notebook_part2.ipynb`**:
   - The second part of the primary notebook, including EDA, preprocessing, model training and evaluation metrics.
     
**Under models directory**
7. **`catboost_model.cbm`**:
   - The first part of the primary notebook, including EDA and some preprocessing.


## Machine Learning Techniques
The **CatBoost Classifier model** was trained and evaluated on the processed data.


## Evaluation Results

      precision   recall   f1-score   support

         0.0       0.96      0.71      0.82     70671
         1.0       0.17      0.68      0.27      6206

    accuracy                           0.71     76877

      macro avg    0.57      0.70      0.55     76877
      weighted avg 0.90      0.71      0.77     76877

**ROC AUC Score: 0.7631627636329994**

