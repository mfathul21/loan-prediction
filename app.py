import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from joblib import dump
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils import load_raw_data, load_data_train_test, plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve, results
from utils import data_transformation, encode_features

def main():
    df = load_raw_data()
    X_train, X_test, y_train, y_test = load_data_train_test()
    class_name = ['Non-Default, Default']

    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            fig = plot_confusion_matrix(y_test, y_test_pred)
            st.pyplot(fig)

        if "Precision Recall Curve" in metrics_list:
            st.subheader("Precision Recall Curve")
            fig = plot_precision_recall_curve(y_test, y_test_pred_proba)
            st.pyplot(fig)

        if "ROC AUC Curve" in metrics_list:
            st.subheader("ROC AUC Curve")
            fig = plot_roc_curve(y_test, y_test_pred_proba)
            st.pyplot(fig)

    st.title("Loan Prediction Based on Customers Behaviour")
    st.markdown("___")

    st.sidebar.subheader("Choose Classifier")
    modelling = st.sidebar.selectbox('Choice Classifier', ['Logistic Regression', 'Decision Tree', 'Random Forest'])

    if modelling == 'Logistic Regression':
        model = LogisticRegression(random_state=42)
        threshold = st.sidebar.slider("Threshold", 0.0, 1.0, value=0.5, step=0.01)
        metric = st.sidebar.multiselect("Metric Other", ["Confusion Matrix", "Precision Recall Curve", "ROC AUC Curve"])

    if modelling == 'Decision Tree':
        model = DecisionTreeClassifier(criterion='entropy', max_features='sqrt', min_samples_split=10, random_state=42)
        threshold = st.sidebar.slider("Threshold", 0.0, 1.0, value=0.5, step=0.01)
        st.sidebar.radio('Criterion', ['gini', 'entropy'])
        st.sidebar.slider("Min Sample Split", 0, 20, step=1)
        metric = st.sidebar.multiselect("Metric Other", ["Confusion Matrix", "Precision Recall Curve", "ROC AUC Curve"])
    
    if modelling == 'Random Forest':
        model = RandomForestClassifier(random_state=42)
        threshold = st.sidebar.slider("Threshold", 0.0, 1.0, value=0.5, step=0.01)
        metric = st.sidebar.multiselect("Metric Other", ["Confusion Matrix", "Precision Recall Curve", "ROC AUC Curve"])
    
    
    if st.sidebar.button("Train Model"):
        model.fit(X_train, y_train)
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_pred_proba >= threshold).astype(int)

        st.subheader(type(model).__name__)
        st.markdown(f'Accuracy {round(accuracy_score(y_test, y_test_pred), 2)}')
        st.markdown(f'Precision {round(precision_score(y_test, y_test_pred), 2)}')
        st.markdown(f'Recall {round(recall_score(y_test, y_test_pred), 2)}')
        st.markdown(f'F1 Score {round(f1_score(y_test, y_test_pred), 2)}')
        st.markdown(f'ROC AUC Score {round(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]), 2)}')
      
        plot_metrics(metric)

    # Create Form Loan Customers
    marital_status = np.vectorize(lambda x: x.title())(df['married/single'].unique())
    house_status = np.vectorize(lambda x: x.replace('_', ' ').title())(df['house_ownership'].unique())
    state_cat = np.append(np.vectorize(lambda x: x.replace('_', ' ').title())(df['state'].unique()), "Other")
    profession_cat = np.append(np.vectorize(lambda x: x.replace('_', ' ').title())(df['profession'].unique()), "Other")

    df_results = results()
    if st.sidebar.checkbox("Apply Loan", False):
        with st.form('Formulir'):
            name = st.text_input("Name")
            birth_date = st.date_input("Birth Date")
            state = st.selectbox("State", state_cat)
            marital = st.selectbox("Marital Status", marital_status)
            salary = st.number_input("Salary per Month (in rupees)", 0)
            cars = st.number_input("Number of Cars", 0)
            house = st.selectbox("House Status", house_status)
            len_house = st.number_input("Length of Time in Current House",0, 100)
            profession = st.selectbox("Profession", profession_cat)
            len_job = st.number_input("Length of Job Experience", 0, 100)
            len_current_job = st.number_input("Length of Current Job Experience", 0, 100)

            current_date = datetime.now()
            age = current_date.year - birth_date.year
            if (current_date.month, current_date.day) < (birth_date.month, birth_date.day):
                age -= 1

            car_ownership = 'yes' if cars > 0 else 'no'
            marital = marital.lower()
            
            submit_button = st.form_submit_button()

            if submit_button:
                df_input = pd.DataFrame({'name':[name], 'age':[age], 'experience':[len_job], 'married/single':[marital], 'car_ownership':[car_ownership], 'profession':[profession],
                                         'state':[state], 'current_job_yrs':[len_current_job], 'current_house_yrs':[len_house],
                                         'income':[salary], 'house_ownership':[house]})
                data_transformation(df_input, test=True)
                encode_features(df_input)
                df_input = pd.get_dummies(df_input, columns=['house_ownership'], drop_first=True)
                st.write(df_input)

    if st.sidebar.checkbox("Show Raw Data", False):
        st.write(f"Shape of data: {df.shape}")
        st.write(df)


# Run the app
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass