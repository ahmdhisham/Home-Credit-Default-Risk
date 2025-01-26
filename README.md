# Home Credit Default Risk
In this project, we aim to develop a machine learning model to predict the likelihood of a customer defaulting on a home credit loan. leveraging past application data to help financial institutions evaluate loan eligibility and mitigate credit risk. This is a classification problem where the target variable is binary, indicating whether the customer will **have difficulties (1)** or **not (0)**.


## Table of Contents
- [Overview](#overview)
- [Objective](#objective)
- [Setup Instructions](#setup-instructions) 
- [Project Structure](#project-structure)
- [Workflow](#workflow)
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


## Setup Instructions
### 1. Clone the Repository
      git clone https://github.com/ahmdhisham/Home-Credit-Default-Risk.git
      cd home-credit-default-risk

### 2. Create and Activate Virtual Environment
      # Create new virtual environemnt     
      python -m venv .venv

      # Activate the virtual environment
      # On Windows:
      .venv\Scripts\activate
      # On macOS/Linux:
      source .venv/bin/activate

### 3. Install Dependencies
      # Install all the required Python packages listed in requirements.txt:
      pip install -r requirements.txt

### 4. Download and Prepare Data
      # Obtain the Home Credit Default Risk dataset from Kaggle.
      # Place the downloaded files in the data directory of the repository. Ensure the following structure:
      home-credit-default-risk/
      ├── data/
      │   ├── application_train.csv
      │   ├── application_test.csv
      │   ├── other_csv_files.csv
      │   ├── other_csv_files.csv

### 5. Run the Project
      # Option 1: Run the Streamlit Web Application
      # Launch the Streamlit app to train models and evaluate metrics:
      streamlit run src/main.py

      # Option 2: Run All Scripts Using the Batch File
      # Use the .bat file to run all preprocessing and training steps:
      src/run_scripts.bat


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

**Under reports directory**

8. **`model_results`**:
   - The images of model evaluation graphs.
9. **`Home_Credit_Dashboard.pbix`**:
   - The dashboard of the business to come out with insights.

**Under data directory (after you follow the setup instructions and download it)**

10. **`columns_description`**:
   - Description of the columns in the dataset folder.
11. **`interim`**:
   - Interim data folder (semi-processed data).
12. **`processed`**:
   - Processed data folder.
13. **`raw`**:
   - raw data folder.


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
        

## Machine Learning Techniques
The **CatBoost Classifier model** was trained and evaluated on the processed data.


## Evaluation Results

      precision   recall   f1-score   support

         0.0       0.96      0.71      0.82     70671
         1.0       0.17      0.68      0.27      6206


      accuracy                         0.71     76877

      macro avg    0.57      0.70      0.55     76877
      weighted avg 0.90      0.71      0.77     76877

**ROC AUC Score: 0.7631627636329994**

