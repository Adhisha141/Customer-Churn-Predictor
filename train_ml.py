"""
train_ml.py
Train an XGBoost classifier on tabular customer features (Telco Churn dataset).
Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

SAVE_PATH = "models/ml_churn_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoders.pkl"


def load_telco_data(csv_path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """
    Load the Kaggle Telco Churn CSV.
    Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    Place in: data/WA_Fn-UseC_-Telco-Customer-Churn.csv
    """
    if not os.path.exists(csv_path):
        print(f"[!] Dataset not found at {csv_path}")
        print("    Using synthetic fallback data for demo purposes.")
        return _synthetic_data()
    return pd.read_csv(csv_path)


def _synthetic_data():
    """Minimal synthetic tabular data for testing."""
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "tenure": np.random.randint(1, 72, n),
        "MonthlyCharges": np.random.uniform(20, 120, n),
        "TotalCharges": np.random.uniform(20, 8000, n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
        "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
        "SeniorCitizen": np.random.randint(0, 2, n),
        "Churn": np.random.choice(["Yes", "No"], n, p=[0.27, 0.73]),
    })
    return df


def preprocess(df):
    df = df.copy()

    # Drop customerID if present
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Fix TotalCharges
    if df["TotalCharges"].dtype == object:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Encode categorical columns
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


def train():
    print("Loading data...")
    df = load_telco_data()
    df, encoders = preprocess(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train_sc, y_train, eval_set=[(X_test_sc, y_test)], verbose=50)

    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, SAVE_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    joblib.dump(list(X.columns), "models/feature_names.pkl")
    print(f"\nModel saved to {SAVE_PATH}")


if __name__ == "__main__":
    train()
