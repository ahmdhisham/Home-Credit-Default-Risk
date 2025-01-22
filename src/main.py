import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from catboost import CatBoostClassifier
from catboost import Pool
from catboost.utils import get_gpu_device_count

import tempfile
import os

# Page configuration
st.set_page_config(
        page_title = 'Home Credit Default Prediction Model',
        layout = 'wide',
        initial_sidebar_state = 'expanded'
        )

# Title
st.title('Home Credit Default Prediction: CatBoost Model Training')

# Upload CSV File
uploaded_file = st.file_uploader("Upload your cleaned CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file, index_col='Unnamed: 0')
    st.write("Preview of uploaded data:")
    st.write(data.head())

    # Print EDA
    st.subheader('Brief EDA')
    st.write("The data is grouped by the gender, showing the percentage of each gender:")
    groupby_target_annuity_avg = data.groupby('CODE_GENDER')['CODE_GENDER'].count()/data.shape[0]
    st.write(groupby_target_annuity_avg)
    st.write("The data is grouped by the target class, showing the count of each class:")
    target_count = data['TARGET'].value_counts()
    st.write(target_count)
    st.bar_chart(target_count)

    # Input fields for hyperparameters
    st.sidebar.header("Model Hyperparameters")
    learning_rate = st.sidebar.number_input("Learning Rate", value=0.1)
    depth = st.sidebar.slider("Depth", min_value=1, max_value=16, value=6)
    iterations = st.sidebar.number_input("Iterations", value=500, step=100)
    test_size = st.sidebar.number_input("Test Size", max_value=0.95, value=0.15, step=0.05)
    task_type = st.sidebar.radio(
    "Select Task Type",
    options=["CPU", "GPU"],
    index=1  # Default to GPU
)

    # Check if a compatible GPU is detected
    if task_type == "GPU" and get_gpu_device_count() == 0:
        st.warning("No compatible GPU detected. Training will fail. Consider switching to 'CPU'.")

    # Train button
    if st.sidebar.button("Train Model"):
        if 'TARGET' not in data.columns:
            st.error("The dataset must contain a 'TARGET' column.")
        else:
            # Create a list of categorical columns
            categorical_cols = list(data.select_dtypes(include=['object']).columns) 

            # Split data
            X = data.drop(columns=['TARGET'])
            y = data['TARGET']

            # Data splitting
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

            # Initialize and train the CatBoost model
            model = CatBoostClassifier(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                cat_features=categorical_cols, 
                auto_class_weights='Balanced',
                random_seed= 42,
                eval_metric='AUC',
                task_type=task_type,
                verbose=0
            )
            model.fit(X_train, y_train)

            # Print after the process is done successfully
            st.sidebar.success("Training of the model has been completed! Please wait for the results and the download link for the model. You can scroll down the page to see results and download your model!")

            # Creating a catboost pool for data
            test_pool = Pool(data=X_test, label=y_test, cat_features=categorical_cols)

            # Get metrics for all iterations
            eval_results = model.eval_metrics(data=test_pool, metrics=['AUC'])

            # Extract every 100th iteration 'AUC' value
            auc_values = eval_results['AUC'][::100]

            # Print training logs
            st.subheader('Training logs')
            st.write(auc_values)

            # Apply model to make predictions
            y_pred = model.predict(X_test)

            # Generate classification report
            report = classification_report(y_test, y_pred)

            # Evaluating the model  
            st.subheader('Model evaluation metrics:')
            st.markdown(f"```plaintext\n{report}\n```")
            st.write("ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

            # Plot the ROC curve for model
            st.subheader('AUC-ROC Curve:')
            model_AUC= roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {model_AUC:.2f})', color='green')
            ax.plot([0, 1], [0, 1], 'r--', label='Random Guess')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax.legend(loc='lower right')
            ax.grid()

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Print training hyperparameters
            st.subheader('Training hyperparameters')
            st.write(model.get_params())

            # Save model
            with tempfile.NamedTemporaryFile(delete=False, suffix=".cbm") as temp_model_file:
                model.save_model(temp_model_file.name)
                temp_model_path = temp_model_file.name
            
            # Provide download link
            with open(temp_model_path, "rb") as file:
                st.download_button(
                    label="Download .cbm Model File",
                    data=file,
                    file_name="trained_model.cbm",
                    mime="application/octet-stream"
                )
            
            # Clean up temp file
            os.unlink(temp_model_path)