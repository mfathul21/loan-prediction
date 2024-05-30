import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

profession_categories = {"Mechanical_engineer": "Engineering", "Chemical_engineer": "Engineering",
                        "Design_Engineer": "Engineering", "Biomedical_Engineer": "Engineering",
                        "Computer_hardware_engineer": "Engineering", "Petroleum_Engineer": "Engineering",
                        "Engineer": "Engineering", "Civil_engineer": "Engineering",
                        "Industrial_Engineer": "Engineering", "Software_Developer": "Technology_and_IT",
                        "Computer_operator": "Technology_and_IT", "Web_designer": "Technology_and_IT",
                        "Technology_specialist": "Technology_and_IT", "Analyst": "Technology_and_IT",
                        "Technical_writer": "Design_and_Creative", "Designer": "Design_and_Creative",
                        "Graphic_Designer": "Design_and_Creative", "Fashion_Designer": "Design_and_Creative",
                        "Architect": "Design_and_Creative", "Artist": "Design_and_Creative",
                        "Financial_Analyst": "Finance_and_Economics", "Economist": "Finance_and_Economics",
                        "Chartered_Accountant": "Finance_and_Economics", "Physician": "Medical_and_Health",
                        "Dentist": "Medical_and_Health", "Surgeon": "Medical_and_Health",
                        "Psychologist": "Medical_and_Health", "Microbiologist": "Medical_and_Health",
                        "Flight_attendant": "Aviation_and_Transportation", "Air_traffic_controller": "Aviation_and_Transportation",
                        "Aviator": "Aviation_and_Transportation", "Civil_servant": "Government_and_Public_Service",
                        "Politician": "Government_and_Public_Service", "Lawyer": "Government_and_Public_Service",
                        "Police_officer": "Government_and_Public_Service", "Firefighter": "Government_and_Public_Service",
                        "Army_officer": "Government_and_Public_Service", "Magistrate": "Government_and_Public_Service",
                        "Hotel_Manager": "Management_and_Administration", "Secretary": "Management_and_Administration",
                        "Consultant": "Management_and_Administration", "Official": "Management_and_Administration",
                        "Scientist": "Science_and_Research", "Geologist": "Science_and_Research",
                        "Statistician": "Science_and_Research", "Surveyor": "Science_and_Research",
                        "Drafter": "Science_and_Research", "Librarian": "Others",
                        "Comedian": "Others", "Chef": "Others", "Technician": "Others"}

state_zone = {'Rajasthan':'West_Zone', 'Maharashtra':'West_Zone',
            'Gujarat':'West_Zone', 'Andhra_Pradesh':'South_Zone',
            'Kerala':'South_Zone', 'Tamil_Nadu':'South_Zone',
            'Puducherry':'South_Zone', 'Telangana':'South_Zone',
            'Karnataka':'South_Zone', 'Jammu_and_Kashmir':'North_Zone',
            'Himachal_Pradesh':'North_Zone', 'Punjab':'North_Zone',
            'Chandigarh':'North_Zone', 'Uttarakhand':'North_Zone',
            'Haryana':'North_Zone', 'Delhi':'North_Zone',
            'Uttar_Pradesh':'North_Zone', 'Bihar':'East_Zone',
            'West_Bengal':'East_Zone', 'Jharkhand':'East_Zone',
            'Odisha':'East_Zone', 'Chhattisgarh':'Central_Zone',
            'Madhya_Pradesh':'Central_Zone', 'Sikkim':'North_East_Zone',
            'Mizoram':'North_East_Zone', 'Manipur':'North_East_Zone',
            'Tripura':'North_East_Zone', 'Assam':'North_East_Zone'}


f_profession = {'Engineering': 44605, 'Government And Public Service': 33735, 'Design And Creative': 29781, 
                'Medical And Health': 25782, 'Science And Research': 25332, 'Technology And It': 24845, 
                'Others': 19386, 'Management And Administration': 19134, 'Aviation And Transportation': 15167, 
                'Finance And Economics': 14233}

f_state = {'South Zone': 68451, 'East Zone': 56886, 'North Zone': 52386, 'West Zone': 46144, 
           'Central Zone': 17956, 'North East Zone': 10177}

len_df = 252000


def load_raw_data():
    df = pd.read_csv('datasets/Training Data.csv', index_col='Id')
    df.columns = [x.lower() for x in df.columns.to_list()]
    return df

@st.cache(persist=True)
def load_data_train_test():
    X_train = pd.read_csv('datasets/X_train.csv') 
    X_test = pd.read_csv('datasets/X_test.csv')
    y_train = pd.read_csv('datasets/y_train.csv')
    y_test = pd.read_csv('datasets/y_test.csv')
    return X_train, X_test, y_train, y_test

def data_cleaning(df):
    df.drop(columns=['city'], axis=1, inplace=True)
    df['state'] = df['state'].replace('Uttar_Pradesh[5]', 'Uttar_Pradesh')
    return df

def data_transformation(df, profession=profession_categories, state=state_zone, test=False):
    if test:
        profession = {key.replace('_', ' ').title(): value for key, value in profession.items()}
        state = {key.replace('_', ' ').title(): value for key, value in state.items()}

    df['profession'] = df['profession'].map(profession).apply(lambda x: x.replace('_', ' ').title())
    df['state'] = df['state'].map(state).apply(lambda x: x.replace('_', ' ').title())

    return df

def encode_features(df, fp=f_profession, fs=f_state, lf=len_df):
    bins = [0, 2500000, 5000000, 7500000, 10000000]
    labels = ['low', 'medium', 'high', 'very high']
    df['income_category'] = pd.cut(df['income'], bins=bins, labels=labels).astype(str)
    df.drop(columns=['income'], axis=1, inplace=True)

    df.loc[:, 'profession'] = df['profession'].map(fp) / lf
    df.loc[:, 'state'] = df['state'].map(fs) / lf
    df.loc[:, 'married/single'] = df.loc[:, 'married/single'].map({'married':1, 'single':0})
    df.loc[:, 'car_ownership'] = df.loc[:, 'car_ownership'].map({'yes':1, 'no':0})
    df.loc[:, 'income_category'] = df.loc[:, 'income_category'].map({'low':0, 'medium':1, 'high':2, 'very high':3}).astype(int)
    df['house_ownership_owned'] = df['house_ownership'].apply(lambda x: int(1) if x == 'owned' else int(0))
    df['house_ownership_rented'] = df['house_ownership'].apply(lambda x: int(1) if x == 'rented' else int(0))

    df.drop(columns=['house_ownership'], inplace=True)
    cat_col = ['married/single', 'car_ownership', 'profession', 'state', 'income_category']
    df[cat_col] = df[cat_col].astype(float)
    return df

def transformation_data_input(df):
    data_transformation(df, test=True)
    encode_features(df)
    return df

def split_to_export(df):
    X = df.drop(columns=['risk_flag'], axis=1)
    y = df['risk_flag']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # smote = SMOTE(random_state=42)
    # X_train, y_train = smote.fit_resample(X_train, y_train)

    # Export dataframe to csv
    X_train.to_csv('datasets/X_train.csv', index=False)
    X_test.to_csv('datasets/X_test.csv', index=False)
    y_train.to_csv('datasets/y_train.csv', index=False)
    y_test.to_csv('datasets/y_test.csv', index=False)
    return X_train, X_test, y_train, y_test

def plot_confusion_matrix(y_true, y_pred, labels=['Non-Default', 'Default']):
    cm = confusion_matrix(y_true, y_pred)

    fig = plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    
    return fig

def plot_precision_recall_curve(y_test, y_test_pred_proba):
    precision, recall, _ = precision_recall_curve(y_test, y_test_pred_proba)
    pr_auc = auc(recall, precision)

    fig = plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR AUC = {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.fill_between(recall, precision, alpha=0.2, color='blue')

    return fig

def plot_roc_curve(y_test, y_test_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.fill_between(fpr, tpr, alpha=0.2, color='darkorange')

    return fig


def metrics_show(metrics_list, y_test, y_test_pred_proba, y_test_pred):
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
