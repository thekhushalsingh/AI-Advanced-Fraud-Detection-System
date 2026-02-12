import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from data_utils import generate_dataset, preprocess
#---------------------------------------------------------

def train_models(data_path="data/transactions.csv", model_dir="models"):
    """Train multiple ML models and save the best one."""
    os.makedirs(model_dir, exist_ok=True)

    # Load / generate data
    if not os.path.exists(data_path):
        print("Dataset not found, generating...")
        generate_dataset(output_path=data_path)

    df = pd.read_csv(data_path)
    df = preprocess(df)

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Save feature columns for prediction consistency
    joblib.dump(list(X_train.columns), os.path.join(model_dir, "feature_columns.pkl"))

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            scale_pos_weight=(len(y_train) - y_train.sum()) / max(y_train.sum(), 1),
            random_state=42, eval_metric="logloss"
        ),
    }

    results = {}
    best_model_name = None
    best_f1 = 0

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        results[name] = {"accuracy": round(acc, 4), "f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4)}
        print(f"  Accuracy: {acc:.4f}  F1: {f1:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}")
        print(classification_report(y_test, y_pred))

        model_path = os.path.join(model_dir, f"{name}.pkl")
        joblib.dump(model, model_path)

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    # Save best model reference
    best_info = {"best_model": best_model_name, "results": results}
    with open(os.path.join(model_dir, "model_info.json"), "w") as f:
        json.dump(best_info, f, indent=2)

    # Copy best model as "best_model.pkl"
    joblib.dump(joblib.load(os.path.join(model_dir, f"{best_model_name}.pkl")),
                os.path.join(model_dir, "best_model.pkl"))

    print(f"\nBest model: {best_model_name} (F1={best_f1:.4f})")
    print(f"Models saved to {model_dir}/")
    return best_model_name, results


if __name__ == "__main__":
    train_models()
