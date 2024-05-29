import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

@st.cache(persist=True)
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

def data_transformation(df, profession, state):
    df['profession'] = df['profession'].map(profession).apply(lambda x: x.replace('_', ' ').title())
    df['state'] = df['state'].map(state).apply(lambda x: x.replace('_', ' ').title())
    return df 

def profession_state_count():
    df = load_raw_data()
    f_profession = df['profession'].value_counts().to_dict()
    f_state = df['state'].value_counts().to_dict()
    len_df = len(df)
    return f_profession, f_state, len_df

def encode_features(df):
    bins = [0, 2500000, 5000000, 7500000, 10000000]
    labels = ['low', 'medium', 'high', 'very high']
    df['income_category'] = pd.cut(df['income'], bins=bins, labels=labels).astype(str)
    df.drop(columns=['income'], axis=1, inplace=True)

    f_profession, f_state, len_df = profession_state_count()

    df.loc[:, 'profession'] = df['profession'].map(f_profession) / len_df
    df.loc[:, 'state'] = df['state'].map(f_state) / len_df
    df.loc[:, 'married/single'] = df.loc[:, 'married/single'].map({'married':1, 'single':0})
    df.loc[:, 'car_ownership'] = df.loc[:, 'car_ownership'].map({'yes':1, 'no':0})
    df.loc[:, 'income_category'] = df.loc[:, 'income_category'].map({'low':0, 'medium':1, 'high':2, 'very high':3}).astype(int)

    df = pd.get_dummies(df, columns=['house_ownership'], drop_first=True)
    cat_col = df.select_dtypes(['object', 'category']).columns
    df[cat_col] = df[cat_col].astype(float)
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

def results():
    df = pd.DataFrame(columns=['name', 'age', 'state', 'married/single', 'income',
                                'car_ownership', 'house_ownership', 'current_house_yrs',
                                'profession', 'experience', 'current_job_yrs'])
    return df
