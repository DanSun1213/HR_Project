import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib



# Load the saved data and model
try:
    with open('employee_attrition_full_data.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    scaler = data_dict['scaler']
    pca = data_dict['pca']
    model = data_dict['best_model']
    feature_names = data_dict['feature_names']
except (FileNotFoundError, KeyError, pickle.UnpicklingError):
    st.error("Error loading pickle file. Attempting to load individual components.")
    try:
        scaler = joblib.load('scaler.joblib')
        pca = joblib.load('pca.joblib')
        model = joblib.load('best_model.joblib')
        with open('feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f]
    except FileNotFoundError:
        st.error("Error loading individual components. Please ensure all required files are present.")
        st.stop()

# Define the top 10 most important features (you should replace these with your actual top features)
top_features = [
    'StockOptionLevel', 'MonthlyIncome', 'EnvironmentSatisfaction', 'TotalWorkingYears', 'YearsAtCompany',
    'JobInvolvement', 'DistanceFromHome', 'Department_Research & Development', 'DailyRate', 'WorkLifeBalance'
]

st.title('Employee Attrition Prediction')

st.write("Please enter the following information about the employee:")

input_data = {}

for feature in top_features:
    if feature == 'OverTime':
        input_data[feature] = 1 if st.checkbox('OverTime') else 0
    elif feature == 'JobRole':
        options = ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 
                   'Manufacturing Director', 'Healthcare Representative', 'Manager', 
                   'Sales Representative', 'Research Director', 'Human Resources']
        selected_role = st.selectbox(f'Select {feature}', options)
        for option in options:
            input_data[f'JobRole_{option}'] = 1 if selected_role == option else 0
    elif feature == 'MaritalStatus':
        options = ['Single', 'Married', 'Divorced']
        selected_status = st.selectbox(f'Select {feature}', options)
        for option in options:
            input_data[f'MaritalStatus_{option}'] = 1 if selected_status == option else 0
    elif feature in ['JobSatisfaction']:
        input_data[feature] = st.slider(f'{feature}', 1, 4, 2)
    else:
        input_data[feature] = st.number_input(f'Enter {feature}', value=0)

from scipy.special import expit  # for the sigmoid function

# ... (rest of the code remains the same)

if st.button('Predict Attrition'):
    # Create a DataFrame with the input data
    input_df = pd.DataFrame([input_data])
    
    # Ensure all feature_names are present in input_data, fill with 0 if missing
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # Reorder columns to match the order used during training
    input_df = input_df[feature_names]
    
    # Scale the input data
    scaled_data = scaler.transform(input_df)
    
    # Apply PCA transformation
    pca_data = pca.transform(scaled_data)
    
    # Make prediction
    prediction = model.predict(pca_data)[0]
    
    # Get decision score
    decision_score = model.decision_function(pca_data)[0]
    
    # Convert decision score to a probability-like score using sigmoid function
    probability_score = expit(decision_score)
    
    # Convert to percentage
    attrition_chance = probability_score * 100
    
    if attrition_chance > 50:
        st.warning(f'High risk of attrition. Chance of leaving: {attrition_chance:.1f}%')
    else:
        st.success(f'Low risk of attrition. Chance of leaving: {attrition_chance:.1f}%')

    # Additional information
    st.info(f"Raw decision score: {decision_score:.2f}")
    st.info("Note: This percentage is an approximation based on the model's decision score.")


