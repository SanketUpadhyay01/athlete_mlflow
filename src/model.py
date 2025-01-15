import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def log_parameters(model, model_name):
    if model_name == "Logistic Regression":
        mlflow.log_param("max_iter", model.max_iter)
    elif model_name == "Random Forest":
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("random_state", model.random_state)
    elif model_name == "Gradient Boosting":
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("learning_rate", model.learning_rate)
        mlflow.log_param("random_state", model.random_state)
    elif model_name == "Decision Tree":
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("random_state", model.random_state)

def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Injured", "Injured"],
                yticklabels=["Not Injured", "Injured"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    artifact_path = f"{model_name}_confusion_matrix.png"
    plt.savefig(artifact_path)
    plt.close()
    mlflow.log_artifact(artifact_path)

def log_and_train_model(model, model_name, X_train, y_train, X_test, y_test, experiment_id):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    with mlflow.start_run(experiment_id=experiment_id, run_name=model_name):
        mlflow.log_param("model_name", model_name)
        log_parameters(model, model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        plot_confusion_matrix(y_test, y_pred, model_name)
        mlflow.sklearn.log_model(model, model_name)

        print(f"Model {model_name} logged to MLflow with accuracy: {accuracy:.4f}")

def main(filepath):
    X, y = preprocess_data(filepath)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    experiment_name = "Collegiate Athlete Injury Analysis"
    try:
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.3, random_state=42),
    }

    for model_name, model in models.items():
        log_and_train_model(model, model_name, X_train, y_train, X_test, y_test, experiment_id)

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    filepath = r"C:\Users\sanket.upadhyay\Desktop\githubaction\athlete_mlflow\data\collegiate_athlete_injury_dataset.csv"
    main(filepath)
