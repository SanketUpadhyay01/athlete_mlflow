import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def load_data(filepath):
    data = pd.read_csv(filepath)
    print(data.head())
    print(data.info())
    return data

def handle_missing_values(data):
    print(f"Missing values before: {data.isnull().sum()}")
    imputer = SimpleImputer(strategy='median')
    for col in data.select_dtypes(include=np.number).columns:
        data[col] = imputer.fit_transform(data[[col]])
    print(f"Missing values after: {data.isnull().sum()}")
    return data

def encode_categorical_features(data):
    cat_columns = data.select_dtypes(include=["object"]).columns
    if not cat_columns.empty:
        data = pd.get_dummies(data, columns=cat_columns, drop_first=True)
        print(f"Categorical values encoded: {list(cat_columns)}")
    else:
        print("No categorical values to encode.")
    return data

def scale_features(data):
    scaler = MinMaxScaler()
    numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data

def preprocess_data(filepath):
    data = load_data(filepath)

    target_column = "Injury_Indicator"
    y = data[target_column]
    X = data.drop(columns=[target_column])

    X = handle_missing_values(X)
    X = encode_categorical_features(X)
    X = scale_features(X)

    print("Preprocessing Complete")
    print(X.head())
    return X, y
