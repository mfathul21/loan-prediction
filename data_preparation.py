import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE

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

def load_raw_data():
    df = pd.read_csv(r'datasets\Training Data.csv', index_col='Id')
    df.columns = [x.lower() for x in df.columns.to_list()]
    return df

def load_data_train_test():
    X_train = pd.read_csv(r'datasets/X_train.csv') 
    X_test = pd.read_csv(r'datasets/X_test.csv')
    y_train = pd.read_csv(r'datasets/y_train.csv')
    y_test = pd.read_csv(r'datasets/y_test.csv')
    return X_train, X_test, y_train, y_test

def data_cleaning(df):
    df.drop(columns=['city'], axis=1, inplace=True)
    df['state'] = df['state'].replace('Uttar_Pradesh[5]', 'Uttar_Pradesh')
    return df

def data_transformation(df, profession, state):
    df['profession'] = df['profession'].map(profession).apply(lambda x: x.replace('_', ' ').title())
    df['state'] = df['state'].map(state).apply(lambda x: x.replace('_', ' ').title())
    return df 

def encode_features(df):
    bins = [0, 2500000, 5000000, 7500000, 10000000]
    labels = ['low', 'medium', 'high', 'very high']
    df['income_category'] = pd.cut(df['income'], bins=bins, labels=labels).astype(str)
    df.drop(columns=['income'], axis=1, inplace=True)

    f_profession = df['profession'].value_counts().to_dict()
    f_state = df['state'].value_counts().to_dict()

    df.loc[:, 'profession'] = df['profession'].map(f_profession) / len(df)
    df.loc[:, 'state'] = df['state'].map(f_state) / len(df)
    df.loc[:, 'married/single'] = df.loc[:, 'married/single'].map({'married':1, 'single':0})
    df.loc[:, 'car_ownership'] = df.loc[:, 'car_ownership'].map({'yes':1, 'no':0})
    df.loc[:, 'income_category'] = df.loc[:, 'income_category'].map({'low':0, 'medium':1, 'high':2, 'very high':3}).astype(int)

    df = pd.get_dummies(df, columns=['house_ownership'], drop_first=True)
    cat_col = df.select_dtypes(['object', 'category']).columns
    df[cat_col] = df[cat_col].astype(float)
    return df

# def data_split(df):
#     X = df.drop(columns=['risk_flag'], axis=1)
#     y = df['risk_flag']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
#     smote = SMOTE(random_state=42)
#     X_train, y_train = smote.fit_resample(X_train, y_train)

#     # Export dataframe to csv
#     X_train.to_csv('datasets/X_train.csv', index=False)
#     X_test.to_csv('datasets/X_test.csv', index=False)
#     y_train.to_csv('datasets/y_train.csv', index=False)
#     y_test.to_csv('datasets/y_test.csv', index=False)
#     return X_train, X_test, y_train, y_test


df = load_raw_data()
df = data_cleaning(df)
df = data_transformation(df, profession_categories, state_zone)
df = encode_features(df)
# X_train, X_test, y_train, y_test = data_split(df)

# print(f'Shape of X_train: {X_train.shape}')
# print(f'Shape of X_test: {X_test.shape}')
# print(f'Shape of y_train: {y_train.shape}')
# print(f'Shape of y_test: {y_test.shape}')