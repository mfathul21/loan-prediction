import datetime as dt 
import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils import load_raw_data, load_data_train_test, transformation_data_input, metrics_show

def main():
    df = load_raw_data()
    X_train, X_test, y_train, y_test = load_data_train_test()

    min_date, max_date = dt.date.today() - dt.timedelta(days=365*80), dt.date.today() - dt.timedelta(days=365*21)
    marital_status = np.vectorize(lambda x: x.title())(df['married/single'].unique())
    house_status = np.vectorize(lambda x: x.replace('_', ' ').title())(df['house_ownership'].unique())
    state_cat = np.vectorize(lambda x: x.replace('_', ' ').title())(df['state'].unique())
    profession_cat = np.vectorize(lambda x: x.replace('_', ' ').title())(df['profession'].unique())

    st.title("Loan Prediction Based on Customers Behaviour 💰")
    st.write("Input your data to assess borrower risk ⬇️⬇️⬇️")
    

    st.sidebar.subheader("Select a Classifier")
    classifier = st.sidebar.selectbox('Choices Classifier', ['Logistic Regression', 'Decision Tree', 'Random Forest'])
    metric = st.sidebar.multiselect("Metric", ["Confusion Matrix", "Precision Recall Curve", "ROC AUC Curve"])
    proba = st.sidebar.radio("Show Probability Default", ["Yes", "No"], index=1, horizontal=True)

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        c = st.sidebar.number_input("C (Inverse of regularization strength)", 0.01, 10.0, step=0.01, value=1.0)
        max_iter = st.sidebar.slider("Max iterations", 100, 500, value=100, step=10)
        model = LogisticRegression(C=c, max_iter=max_iter, random_state=42)
    if classifier == "Decision Tree":
        st.sidebar.subheader("Model Hyperparameters")
        criterion = st.sidebar.selectbox("Criterion", ("gini", "entropy"))
        max_features = st.sidebar.selectbox("Max features", ("sqrt", "log2", None))
        min_samples_split = st.sidebar.slider("Min samples split", 2, 20, value=10, step=1)
        model = DecisionTreeClassifier(criterion=criterion, max_features=max_features, min_samples_split=min_samples_split, random_state=42)
    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of trees in the forest", 10, 500, step=10, value=100)
        max_features = st.sidebar.selectbox("Max features", ("sqrt", "log2", None), index=0)
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", (True, False))
        model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, bootstrap=bootstrap, random_state=42)
        
    threshold = st.sidebar.slider("Threshold (Recomenndation: 0.30)", 0.0, 1.0, value=0.5, step=0.01)

    st.sidebar.markdown("___")
    st.sidebar.subheader("Classifier Parameter Recommendation")
    st.sidebar.write("classifier: RandomForestClassifier")
    st.sidebar.write("n_estimators: 100")
    st.sidebar.write("max_features: sqrt")
    st.sidebar.write("bootstrap=True")
    

    with st.form('Formulir'):
        name = st.text_input("Name")
        birth_date = st.date_input("Birth Date", value=max_date, min_value=min_date, max_value=max_date)
        state = st.selectbox("State", state_cat)
        marital = st.selectbox("Marital Status", marital_status)
        income = st.number_input("Income per Month in Rupees (not null)", 1500, max_value=15000000)
        cars = st.number_input("Number of Cars", 0)
        house = st.selectbox("House Status", house_status)
        len_house = st.number_input("Length of Time in Current House",0, 100)
        profession = st.selectbox("Profession", profession_cat)
        len_job = st.number_input("Length of Job Experience", 0, 100)
        len_current_job = st.number_input("Length of Current Job Experience", 0, 100)

        current_date = dt.datetime.now()
        age = current_date.year - birth_date.year
        if (current_date.month, current_date.day) < (birth_date.month, birth_date.day):
            age -= 1

        car_ownership = 'yes' if cars > 0 else 'no'
        marital = marital.lower()
        house = house.lower().replace(' ','')
        
        submit_button = st.form_submit_button()

        if submit_button:
            df_input = pd.DataFrame({'name':[name], 'age':[age], 'experience':[len_job], 
                                     'married/single':[marital], 'car_ownership':[car_ownership], 'profession':[profession], 
                                     'state':[state], 'current_job_yrs':[len_current_job], 'current_house_yrs':[len_house],
                                     'income':[income], 'house_ownership':[house]})          

            transformation_data_input(df_input)
            model.fit(X_train, y_train)
            X = df_input.drop(columns=['name'], axis=1)
            y_pred_proba = model.predict_proba(X)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)


            col1, col2 = st.columns([1,1])
            with col1:
                st.subheader(f"{'Loan Accepted' if y_pred[0] == 0 else 'Loan Rejected'}")
                st.write(f"**Username**: {name.title()}")
                if proba == "Yes":
                    st.write(f"**Probability Default**: {y_pred_proba[0]}")
                st.write(f"**Age**: {age}")
                st.write(f"**Status**: {marital.title()}")   
                st.write(f"**Profession**: {profession}")  
                st.write(f"**Income**: ₹{income}")  

            with col2:      
                y_test_pred_proba = model.predict_proba(X_test)[:, 1]
                y_test_pred = (y_test_pred_proba >= threshold).astype(int)

                st.subheader("Model Evaluation")
                st.markdown(f'**Accuracy**: {round(accuracy_score(y_test, y_test_pred), 2)}')
                st.markdown(f'**Precision**: {round(precision_score(y_test, y_test_pred), 2)}')
                st.markdown(f'**Recall**: {round(recall_score(y_test, y_test_pred), 2)}')
                st.markdown(f'**F1 Score**: {round(f1_score(y_test, y_test_pred), 2)}')
                st.markdown(f'**ROC AUC Score**: {round(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]), 2)}')

            metrics_show(metric, y_test, y_test_pred_proba, y_test_pred)    
    

# Run the app
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass