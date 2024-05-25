import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
from utils import load_raw_data, data_cleaning, data_transformation, encode_features, split_to_export


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


df = load_raw_data()
df = data_cleaning(df)
df = data_transformation(df, profession_categories, state_zone)
df = encode_features(df)
X_train, X_test, y_train, y_test = split_to_export(df)

print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of y_test: {y_test.shape}')