import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
from utils import load_raw_data, data_cleaning, data_transformation, encode_features, split_to_export



df = load_raw_data()
df = data_cleaning(df)
df = data_transformation(df)
df = encode_features(df)
X_train, X_test, y_train, y_test = split_to_export(df)

print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of y_test: {y_test.shape}')