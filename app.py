import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler 
from preprocessing import load_data_train, data_preparation


marital_status = ['Married', 'Single']

house_status = ['Owned', 'Rented', "Don't own"]

profession_cat = ['Mechanical Engineer', 'Software Developer', 'Technical Writer',
            'Civil Servant', 'Librarian', 'Economist', 'Flight Attendant', 'Architect',
            'Designer', 'Physician', 'Financial Analyst', 'Air Traffic Controller',
            'Politician', 'Police Officer', 'Artist', 'Surveyor', 'Design Engineer',
            'Chemical Engineer', 'Hotel Manager', 'Dentist', 'Comedian',
            'Biomedical Engineer', 'Graphic Designer', 'Computer Hardware Engineer',
            'Petroleum Engineer', 'Secretary', 'Computer Operator',
            'Chartered Accountant', 'Technician', 'Microbiologist', 'Fashion Designer',
            'Aviator' 'Psychologist', 'Magistrate', 'Lawyer', 'Firefighter', 'Engineer',
            'Official', 'Analyst', 'Geologist', 'Drafter', 'Statistician', 'Web_designer',
            'Consultant', 'Chef', 'Army Officer', 'Surgeon', 'Scientist', 'Civil Engineer',
            'Industrial Engineer', 'Technology Specialist']

state_cat = ['Madhya Pradesh', 'Maharashtra', 'Kerala', 'Odisha', 'Tamil Nadu', 'Gujarat',
        'Rajasthan', 'Telangana', 'Bihar', 'Andhra Pradesh', 'West Bengal', 'Haryana',
        'Puducherry', 'Karnataka', 'Uttar Pradesh', 'Himachal Pradesh', 'Punjab',
        'Tripura', 'Uttarakhand', 'Jharkhand', 'Mizoram', 'Assam', 'Jammu and Kashmir',
        'Delhi', 'Chhattisgarh', 'Chandigarh', 'Manipur', 'Sikkim', 'Other']

def main():
    st.title("Loan Prediction Based on Customers Behaviour")

    df = load_data_train()
    X_train, X_test, y_train, y_test = data_preparation(df)

    modelling = st.sidebar.selectbox('Choice Classifier', ['Logistic Regression', 'Decision Tree', 'Random Forest'])
    oversampling = st.sidebar.selectbox('Oversampling Method', ['RandomOversampling', 'SMOTE'])
    threshold = st.sidebar.slider("Threshold", 0.0, 1.0, value=0.5, step=0.01)
    
    if modelling == 'Logistic Regression':
        model = LogisticRegression(random_state=42)
    elif modelling == 'Decision Tree':
        model = DecisionTreeClassifier(criterion='entropy', max_features='auto', min_samples_split=10, random_state=42)
    elif modelling == 'Random Forest':
        model = RandomForestClassifier(random_state=42)
    
    if oversampling == 'RandomOverSampling':
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
    elif oversampling == 'SMOTE':
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    if st.sidebar.button("Train Model"):
        model.fit(X_train, y_train)
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_pred_proba >= threshold).astype(int)

        st.subheader(type(model).__name__)
        st.markdown(f'Accuracy {accuracy_score(y_test, y_test_pred)}')
        st.markdown(f'Precision {precision_score(y_test, y_test_pred)}')
        st.markdown(f'Recall {recall_score(y_test, y_test_pred)}')
        st.markdown(f'F1 Score {f1_score(y_test, y_test_pred)}')
        st.markdown(f'ROC AUC Score {roc_auc_score(y_test, y_test_pred)}')

        metric = st.radio("Metric Other", ["Confusion Matrix", "Precision Recall Curve"], index=1, horizontal=True)

    # Create Form Loan Customers
    if st.sidebar.checkbox("Apply Loan", False):
        with st.form('Formulir'):
            name = st.text_input("Name")
            birth_date = st.date_input("Birth Date")
            state = st.selectbox("State", state_cat)
            if state == "Other":
                state = st.text_input("Please specify your state")
            marital = st.selectbox("Marital Status", marital_status)
            salary = st.number_input("Salary per Month (in rupees)", 0)
            cars = st.number_input("Number of Cars", 0)
            house = st.selectbox("House Status", house_status)
            len_house = st.number_input("Length of Time in Current House",0, 100)
            profession = st.selectbox("Profession", profession_cat)
            len_job = st.number_input("Length of Job Experience", 0, 100)
            len_current_job = st.number_input("Length of Current Job Experience", 0, 100)
            
            if profession == "Other":
                profession = st.text_input("Please specify your profession")
            
            submit_button = st.form_submit_button()

    if st.sidebar.checkbox("Show Raw Data", False):
        st.write(f"Shape of data: {df.shape}")
        st.write(df)
    


# Run the app
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass

