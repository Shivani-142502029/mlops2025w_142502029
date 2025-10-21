# src/train.py
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore_data():
    """Load and explore the Iris dataset"""
    print("Loading Iris dataset...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='species')
    df = X.copy()
    df['species'] = y
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    print(f"Dataset shape: {df.shape}")
    return X, y, df

def preprocess_data(X, y, test_size=0.2, random_state=42):
    """Split and preprocess the data"""
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    print(f"Training set size: {X_train_scaled.shape[0]}")
    print(f"Test set size: {X_test_scaled.shape[0]}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }

def train_random_forest(X_train, y_train, X_test, y_test, **params):
    """Train Random Forest model and log to an EXISTING MLflow run"""
    mlflow.log_params(params)
    mlflow.log_param("model_type", "RandomForestClassifier")

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    signature = infer_signature(X_test, y_pred_test)
    
    # This is the key change: log and register the model in one step
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_test.head(3),
        registered_model_name="iris-classifier"  # Register the model
    )
    
    print(f"  Random Forest - Test Accuracy: {test_metrics['accuracy']:.4f}")
    return model, test_metrics['accuracy']

def train_logistic_regression(X_train, y_train, X_test, y_test, **params):
    """Train Logistic Regression model and log to an EXISTING MLflow run"""
    mlflow.log_params(params)
    mlflow.log_param("model_type", "LogisticRegression")
    
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    signature = infer_signature(X_test, y_pred_test)
    
    # This is the key change: log and register the model in one step
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_test.head(3),
        registered_model_name="iris-classifier" # Register the model
    )
    
    print(f"  Logistic Regression - Test Accuracy: {test_metrics['accuracy']:.4f}")
    return model, test_metrics['accuracy']

def hyperparameter_tuning():
    """Perform hyperparameter tuning, automatically registering the best model."""
    X, y, df = load_and_explore_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    mlflow.set_experiment("Iris Classification Hyperparameter Tuning")

    with mlflow.start_run(run_name="Hyperparameter Tuning Session"):
        best_accuracy = -1.0 # Initialize to a value lower than any possible accuracy

        rf_params = [
            {'n_estimators': 50, 'max_depth': 3, 'random_state': 42},
            {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
        ]
        
        print("\nTraining Random Forest variants...")
        for params in rf_params:
            with mlflow.start_run(nested=True, run_name=f"RF_{params['n_estimators']}_{params.get('max_depth', 'None')}"):
                model, accuracy = train_random_forest(X_train, y_train, X_test, y_test, **params)
                if accuracy > best_accuracy:
                    print(f"    New best model found! Accuracy: {accuracy:.4f}")
                    best_accuracy = accuracy

        lr_params = [
            {'C': 1.0, 'random_state': 42, 'max_iter': 1000},
            {'C': 10.0, 'random_state': 42, 'max_iter': 1000},
        ]

        print("\nTraining Logistic Regression variants...")
        for params in lr_params:
            with mlflow.start_run(nested=True, run_name=f"LR_C_{params['C']}"):
                model, accuracy = train_logistic_regression(X_train, y_train, X_test, y_test, **params)
                if accuracy > best_accuracy:
                    print(f"    New best model found! Accuracy: {accuracy:.4f}")
                    best_accuracy = accuracy

        mlflow.log_metric("best_accuracy", best_accuracy)
        print(f"\nHyperparameter tuning complete. Best accuracy: {best_accuracy:.4f}")

def load_and_predict(model_name):
    """Load latest version of the model and make predictions."""
    print(f"\nLoading latest version of model '{model_name}' from registry...")
    
    # The client helps us find the latest version number
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
    
    print(f"Loading version {latest_version}...")
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest_version}")
    
    X, _, _ = load_and_explore_data()
    X_sample = X.head(5)
    
    predictions = model.predict(X_sample)
    
    results = pd.DataFrame(X_sample)
    results['predicted_class'] = [ ['setosa', 'versicolor', 'virginica'][p] for p in predictions]

    print("\nPrediction Results:")
    print(results.to_string())

def main():
    """Main function to run the complete pipeline"""
    print("=== MLflow Iris Classification Pipeline ===\n")
    
    model_name = "iris-classifier"
    
    # This function now handles training and registering the best model version
    hyperparameter_tuning()

    # This function loads the latest registered version and predicts
    load_and_predict(model_name)

    print(f"\n=== Pipeline Completed Successfully ===")
    print(f"To see your results, run 'mlflow ui' in your terminal.")

if __name__ == "__main__":
    main()

