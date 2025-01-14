import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

filepath = r"athlete_mlflow\data\collegiate_athlete_injury_dataset.csv"

def load_data(filepath):
    # Load the data from the file
    data = pd.read_csv(filepath)
    print(data.head())
    print(data.info())

    return data

def handle_missing_values(data):
    # Check for missing values in:
    print(f"Missing values before: {data.isnull().sum()}")
    imputer = SimpleImputer(strategy='median')
    for col in data.select_dtypes(include = np.number).columns:
        data[col] = imputer.fit_transform(data[[col]])
    print(f"Missing values after: {data.isnull().sum()}")
    return data

def encode_categorical_features(data):
    # One-hot encoding for categorical features
    cat_columns = data.select_dtypes(include = ["object"]).columns
    if not cat_columns.empty:
        data = pd.get_dummies(data, columns = cat_columns, drop_first=True)
        print(f"Categorical values encoded: {list(cat_columns)}")
    else:
        print("No cateorical values to encode")
    return data

def normalise_features(data):
    # Normalise the features using z-score
    scaler = StandardScaler()
    numeric_columns = data.select_dtypes(include = np.number).columns
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data

def scale_features(data):
    # Scale the features using Min-Max Scaler
    scaler = MinMaxScaler()
    numeric_columns = data.select_dtypes(include = np.number).columns
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data

def preprocess_data(filepath):
    data = load_data(filepath)
    data = handle_missing_values(data)
    data = encode_categorical_features(data)
    data = normalise_features(data)
    data = scale_features(data)
    print("Preprocessing Complete")
    print(data.head())
    return data
