from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.feature_engineering import preprocess_data
import pandas as pd

FILEPATH = "data/collegiate_athlete_injury_dataset.csv"
TARGET_COLUMN = "Injury_Indicator"  


def test_logistic_regression():
    df = preprocess_data(FILEPATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    assert accuracy >= 0.7, "Logistic Regression accuracy is below 0.7."
    assert precision >= 0.7, "Logistic Regression precision is below 0.7."
    assert recall >= 0.7, "Logistic Regression recall is below 0.7."


def test_random_forest():
    df = preprocess_data(FILEPATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    assert accuracy >= 0.75, "Random Forest accuracy is below 0.75."
    assert precision >= 0.75, "Random Forest precision is below 0.75."
    assert recall >= 0.75, "Random Forest recall is below 0.75."


def test_decision_tree():
    df = preprocess_data(FILEPATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    assert accuracy >= 0.65, "Decision Tree accuracy is below 0.65."
    assert precision >= 0.65, "Decision Tree precision is below 0.65."
    assert recall >= 0.65, "Decision Tree recall is below 0.65."


def test_gradient_boosting():
    df = preprocess_data(FILEPATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    assert accuracy >= 0.75, "Gradient Boosting accuracy is below 0.75."
    assert precision >= 0.75, "Gradient Boosting precision is below 0.75."
    assert recall >= 0.75, "Gradient Boosting recall is below 0.75."
