import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def load_and_prepare_data(file_path, target_column):
    """Load dataset and prepare features for modeling."""
    df = pd.read_csv(file_path)

    # Drop irrelevant columns
    df = df.drop(columns=['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId'])

    # Ensure no missing values
    df = df.fillna(0)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y

def split_data(X, y):
    """Split the data into train and test sets."""
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_models(X_train, y_train):
    """Train Logistic Regression and Random Forest models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate models on test data."""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
        }
    return results

def save_best_model(models, results, output_dir):
    """Save the best-performing model."""
    best_model_name = max(results, key=lambda x: results[x]['Accuracy'])
    best_model = models[best_model_name]
    joblib.dump(best_model, f"{output_dir}/{best_model_name.replace(' ', '_')}.pkl")
    return best_model_name

def main(data_path, target_column, output_dir):
    """Main function for training and evaluating models."""
    X, y = load_and_prepare_data(data_path, target_column)
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    
    print("Model Evaluation Results:")
    for model, metrics in results.items():
        print(f"\n{model}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    best_model = save_best_model(models, results, output_dir)
    print(f"\nBest model '{best_model}' has been saved to {output_dir}.")

if __name__ == "__main__":
    # Example usage
    data_path = "preprocessed_dataset.csv"  # Replace with your Colab dataset path
    target_column = "FraudResult"
    output_dir = "models"  # Save models in the Colab workspace
    main(data_path, target_column, output_dir)
