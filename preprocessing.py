import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

def load_data_train():
    df = pd.read_csv(r'C:\Users\ASUS\myProject\loan-prediction\datasets\Training Data.csv', index_col='Id')
    df.columns = [x.lower() for x in df.columns.to_list()]
    return df

def handle_missing_values(df):
    for col in df.columns:
        percentage_missing = df[col].isna().sum() / len(df)

        if percentage_missing == 0:
            pass
        elif percentage_missing <= 0.1:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df.drop(columns=[col], inplace=True)
    return df


def handle_duplicates(df):
    df.drop_duplicates(inplace=True)
    return df


def data_preparation(df):
    # Replace Uttar_Pradesh[5] to Uttar_Pradesh
    df['state'] = df['state'].replace('Uttar_Pradesh[5]', 'Uttar_Pradesh')

    # Create New Feature 'income_category'
    bins = [0, 2500000, 5000000, 7500000, 10000000]
    labels = ['low', 'medium', 'high', 'very high']
    df['income_category'] = pd.cut(df['income'], bins=bins, labels=labels)

    # Drop Columns 'income' and 'city'
    df.drop(columns=['income', 'city'], axis=1, inplace=True)

    # Mapping Profession Categories
    profession_categories = {
    "Mechanical_engineer": "Engineering", "Chemical_engineer": "Engineering",
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
    
    df['profession'] = df['profession'].map(profession_categories)

    # Mapping State Categories
    state_zone = {
    'Rajasthan':'West_Zone', 'Maharashtra':'West_Zone',
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

    df['state'] = df['state'].map(state_zone)    

    # Encoding Categorical Features
    frequency_profession = df['profession'].value_counts().to_dict()
    frequency_state = df['state'].value_counts().to_dict()

    df.loc[:, 'profession'] = df['profession'].map(frequency_profession) / len(df)
    df.loc[:, 'state'] = df['state'].map(frequency_state) / len(df)
    df.loc[:, 'married/single'] = df.loc[:, 'married/single'].map({'married':1, 'single':0})
    df.loc[:, 'car_ownership'] = df.loc[:, 'car_ownership'].map({'yes':1, 'no':0})
    df.loc[:, 'income_category'] = df.loc[:, 'income_category'].map({'low':0, 'medium':1, 'high':2, 'very high':3})

    df = pd.get_dummies(df, columns=['house_ownership'], drop_first=True)
    cat_col = ['married/single', 'car_ownership', 'income_category', 'profession',
                'state', 'house_ownership_owned','house_ownership_rented']

    df[cat_col] = df[cat_col].astype(float)

    y = df['risk_flag']
    X = df.drop(columns=['risk_flag'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
 
    return X_train, X_test, y_train, y_test

