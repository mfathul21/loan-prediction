import streamlit as st 
import datetime
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils import load_raw_data, load_data_train_test
from utils import transformation_data_input, metrics_show

def main():
    df = load_raw_data()
    X_train, X_test, y_train, y_test = load_data_train_test()
    class_name = ['Non-Default, Default']

    min_date, max_date = datetime.date.today() - datetime.timedelta(days=365*80), datetime.date.today() - datetime.timedelta(days=365*21)
    marital_status = np.vectorize(lambda x: x.title())(df['married/single'].unique())
    house_status = np.vectorize(lambda x: x.replace('_', ' ').title())(df['house_ownership'].unique())
    state_cat = np.vectorize(lambda x: x.replace('_', ' ').title())(df['state'].unique())
    profession_cat = np.vectorize(lambda x: x.replace('_', ' ').title())(df['profession'].unique())

    df_form = pd.DataFrame(['name', 'age', 'experience', 'married/single', 'car_ownership', 'profession',
                            'state', 'current_job_yrs', 'current_house_yrs', 'income', 'house_ownership'])


    st.title("Loan Prediction Based on Customers Behaviour 💰")
    st.write("Input your data to assess borrower risk ⬇️⬇️⬇️")

    app = st.sidebar.selectbox("Appplication Model", ["Existing Model", "Re-Train Model"])
    
    if app == "Existing Model":
        model = load('best_model.joblib')
        eval = st.sidebar.selectbox("Show Evaluation Model", ["None", "Sure"])
        
    if app == "Re-Train Model":
        classifier = st.sidebar.selectbox('Choices Classifier', ['Logistic Regression', 'Decision Tree', 'Random Forest'])
        

    if eval == "Sure":
        metric = st.sidebar.multiselect("Metric", ["Confusion Matrix", "Precision Recall Curve", "ROC AUC Curve"])

    threshold = st.sidebar.slider("Threshold (Recomenndation: 0.30)", 0.0, 1.0, value=0.5, step=0.01)
    with st.form('Formulir'):
        name = st.text_input("Name")
        birth_date = st.date_input("Birth Date", value=max_date, min_value=min_date, max_value=max_date)
        state = st.selectbox("State", state_cat)
        marital = st.selectbox("Marital Status", marital_status)
        income = st.number_input("Income per Month in Rupees (not null)", 0)
        cars = st.number_input("Number of Cars", 0)
        house = st.selectbox("House Status", house_status)
        len_house = st.number_input("Length of Time in Current House",0, 100)
        profession = st.selectbox("Profession", profession_cat)
        len_job = st.number_input("Length of Job Experience", 0, 100)
        len_current_job = st.number_input("Length of Current Job Experience", 0, 100)

        current_date = datetime.datetime.now()
        age = current_date.year - birth_date.year
        if (current_date.month, current_date.day) < (birth_date.month, birth_date.day):
            age -= 1

        car_ownership = 'yes' if cars > 0 else 'no'
        marital = marital.lower()
        house = house.lower().replace(' ','')
        
        submit_button = st.form_submit_button()

        if submit_button:
            df_input = pd.DataFrame({'name':[name], 'age':[age], 'experience':[len_job], 'married/single':[marital], 
                                     'car_ownership':[car_ownership], 'profession':[profession], 'state':[state], 
                                     'current_job_yrs':[len_current_job], 'current_house_yrs':[len_house],
                                     'income':[income], 'house_ownership':[house]})          

            transformation_data_input(df_input)
            X = df_input.drop(columns=['name'], axis=1)
            y_pred_proba = model.predict_proba(X)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)

            col1, col2 = st.columns([1,1])
            with col1:
                st.subheader(f"{'Good Borrower' if y_pred[0] == 0 else 'Bad Borrower'}\n")
                st.write(f"**Username**: {name.title()}")
                st.write(f"**Age**: {age}")
                st.write(f"**Status**: {marital.title()}")   
                st.write(f"**Profession**: {profession}")  
                st.write(f"**Income**: ₹{income}")  

            if eval == "Sure":
                with col2:               
                    model.fit(X_train, y_train)
                    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
                    y_test_pred = (y_test_pred_proba >= threshold).astype(int)

                    st.subheader("Model Evaluation")
                    st.markdown(f'**Accuracy**: {round(accuracy_score(y_test, y_test_pred), 2)}')
                    st.markdown(f'**Precision**: {round(precision_score(y_test, y_test_pred), 2)}')
                    st.markdown(f'**Recall**: {round(recall_score(y_test, y_test_pred), 2)}')
                    st.markdown(f'**F1 Score**: {round(f1_score(y_test, y_test_pred), 2)}')
                    st.markdown(f'**ROC AUC Score**: {round(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]), 2)}')

                metrics_show(metric, y_test, y_test_pred_proba, y_test_pred)
        
    if app == "Show Data":
        st.write(df_form)
    
    

# Run the app
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass


# st.sidebar.write('**Excellent:** Sangat rendah risiko default.')
# st.sidebar.write('**Very Good:** Rendah risiko default.')
# st.sidebar.write('**Good:** moderat risiko default.')
# st.sidebar.write('**Fair:** Sedikit lebih berisiko.')
# st.sidebar.write('**Poor:** Tinggi risiko default.')