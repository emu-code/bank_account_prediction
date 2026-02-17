import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

st.set_page_config(page_title="Financial Inclusion Analysis", layout="wide")

st.title("Financial Inclusion Analysis")
st.markdown("Predicting Bank Account Access for Financial Inclusion")

# Load saved models
with open('models.pkl', 'rb') as f:
    models = pickle.load(f)

df = pd.read_csv('Financial_inclusion_dataset.csv')
df.drop(columns=['uniqueid', 'year'], inplace=True)

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'bank_account' in num_cols:
    num_cols.remove('bank_account')

# Sidebar Navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select a page:", 
                    ["Dashboard", "Make Prediction", "Model Comparison", "About"])

if page == "Dashboard":
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Individuals", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        has_account = (df['bank_account'] == 'Yes').sum()
        st.metric("With Bank Account", has_account)
    with col4:
        no_account = (df['bank_account'] == 'No').sum()
        st.metric("Without Bank Account", no_account)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Bank Account Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        account_counts = df['bank_account'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        ax.pie(account_counts.values, labels=account_counts.index, autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax.set_title("Financial Inclusion Rate")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Data Sample")
        st.dataframe(df.head(10), use_container_width=True)
    

elif page == "Make Prediction":
    st.header("Predict Bank Account Access")
    st.markdown("Enter demographics and financial information to predict bank account eligibility")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Enter Your Information")
        input_data = {}
        
        for col in df.columns:
            if col == 'bank_account':
                continue
            
            if col in cat_cols:
                unique_vals = sorted(df[col].dropna().unique().tolist())
                input_data[col] = st.selectbox(f"{col}", unique_vals, key=f"select_{col}")
            else:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                input_data[col] = st.slider(f"{col}", min_val, max_val, mean_val, key=f"slider_{col}")
    
    with col2:
        st.subheader("Prediction Results")
        st.write("")
        st.write("")
        
        if st.button("Predict Bank Account Access", key="predict_btn", use_container_width=True):
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical
            for col in cat_cols:
                if col in models['le_dict']:
                    input_df[col] = models['le_dict'][col].transform(input_df[col].astype(str))
            
            # Scale numerical
            input_df[num_cols] = models['scaler'].transform(input_df[num_cols])
            
            # Get predictions
            log_pred = models['log_reg'].predict(input_df)[0]
            rf_pred = models['rf'].predict(input_df)[0]
            xgb_pred = models['xgb'].predict(input_df)[0]
            
            st.success("Prediction Complete!")
            st.write("")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                result_color = "green" if log_pred == "Yes" else "red"
                st.success(f"Logistic Regression\n{log_pred}")
            
            with col_res2:
                result_color = "green" if rf_pred == "Yes" else "red"
                st.success(f"Random Forest\n{rf_pred}")
            
            with col_res3:
                result_color = "green" if xgb_pred == "Yes" else "red"
                st.success(f"XGBoost\n{xgb_pred}")

elif page == "Model Comparison":
    st.header("Model Performance Analysis")
    
    Y_pred_log = models['Y_pred_log']
    Y_pred_rf = models['Y_pred_rf']
    Y_pred_xgb = models['Y_pred_xgb']
    Y_test = models['Y_test']
    
    # Calculate accuracies
    acc_log = (Y_pred_log == Y_test).mean()
    acc_rf = (Y_pred_rf == Y_test).mean()
    acc_xgb = (Y_pred_xgb == Y_test).mean()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Logistic Regression Accuracy", f"{acc_log:.2%}")
    
    with col2:
        st.metric("Random Forest Accuracy", f"{acc_rf:.2%}")
    
    with col3:
        st.metric("XGBoost Accuracy", f"{acc_xgb:.2%}")
    

elif page == "About":
    st.header("About Financial Inclusion")
    
    st.markdown("""
    ### What is Financial Inclusion?
    Financial inclusion is the availability and equality of opportunities to access 
    financial services. This project aims to identify individuals who lack access to 
    bank accounts and predict which demographic and economic factors influence 
    banking accessibility.
    
    ### Project Goal
    By analyzing demographic and financial data, we can:
    - Identify underbanked populations
    - Predict who is likely to have/need a bank account
    - Help financial institutions reach excluded populations
    - Support policy-making for financial inclusion initiatives
    
    ### Models Used
    We trained and compared three machine learning models:
    
    **1. Logistic Regression**
    - Baseline model for binary classification
    - Highly interpretable
    - Fast and efficient
    
    **2. Random Forest**
    - Ensemble learning approach
    - Good predictive power with interpretability
    - Handles non-linear relationships
    
    **3. XGBoost**
    - Gradient boosting technique
    - Best predictive performance
    - Handles imbalanced data well
    
    ### Data Processing
    - Categorical variables encoded using LabelEncoder
    - Numerical variables scaled using StandardScaler
    - Train-test split: 80-20
    - Class weights balanced to handle imbalanced data
    
    ### How to Use This App
    1. **Dashboard:** Explore the overall dataset and statistics
    2. **Make Prediction:** Input individual characteristics to predict bank account access
    3. **Model Comparison:** Compare performance of all three models
    4. **About:** Learn more about financial inclusion and the project
    """)