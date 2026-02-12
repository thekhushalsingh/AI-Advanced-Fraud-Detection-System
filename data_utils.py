import pandas as pd
import numpy as np
import os

def generate_dataset(n_samples=10000, fraud_ratio=0.05, output_path="data/transactions.csv"):
    """Generate a synthetic fraud detection dataset."""
    np.random.seed(42)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    legit = pd.DataFrame({
        "amount": np.random.lognormal(mean=4, sigma=1, size=n_legit).round(2),
        "hour_of_day": np.random.randint(6, 23, size=n_legit),
        "day_of_week": np.random.randint(0, 7, size=n_legit),
        "merchant_category": np.random.choice(["grocery", "electronics", "clothing", "restaurant", "gas", "travel", "online"], size=n_legit),
        "transaction_type": np.random.choice(["card_present", "online", "contactless"], size=n_legit, p=[0.5, 0.3, 0.2]),
        "distance_from_home": np.abs(np.random.normal(10, 15, size=n_legit)).round(1),
        "distance_from_last_txn": np.abs(np.random.normal(5, 10, size=n_legit)).round(1),
        "avg_daily_transactions": np.random.poisson(3, size=n_legit).astype(float),
        "velocity_last_hour": np.random.poisson(1, size=n_legit).astype(float),
        "failed_attempts_24h": np.random.choice([0, 0, 0, 0, 1], size=n_legit),
        "account_age_days": np.random.randint(30, 3650, size=n_legit),
        "is_international": np.random.choice([0, 0, 0, 0, 1], size=n_legit),
        "is_fraud": 0,
    })

    # Fraudulent transactions
    fraud = pd.DataFrame({
        "amount": np.random.lognormal(mean=6, sigma=1.5, size=n_fraud).round(2),
        "hour_of_day": np.random.choice([0, 1, 2, 3, 4, 5, 23, 22], size=n_fraud),
        "day_of_week": np.random.randint(0, 7, size=n_fraud),
        "merchant_category": np.random.choice(["electronics", "travel", "online", "jewelry"], size=n_fraud),
        "transaction_type": np.random.choice(["online", "card_present"], size=n_fraud, p=[0.7, 0.3]),
        "distance_from_home": np.abs(np.random.normal(200, 150, size=n_fraud)).round(1),
        "distance_from_last_txn": np.abs(np.random.normal(300, 200, size=n_fraud)).round(1),
        "avg_daily_transactions": np.random.poisson(8, size=n_fraud).astype(float),
        "velocity_last_hour": np.random.poisson(5, size=n_fraud).astype(float),
        "failed_attempts_24h": np.random.choice([1, 2, 3, 4, 5], size=n_fraud),
        "account_age_days": np.random.randint(1, 60, size=n_fraud),
        "is_international": np.random.choice([0, 1, 1, 1], size=n_fraud),
        "is_fraud": 1,
    })

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TXN{str(i).zfill(6)}" for i in range(len(df))])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} transactions ({n_fraud} fraud, {n_legit} legit) -> {output_path}")
    return df

#------------------------------------------------
def preprocess(df):
    """Clean and encode the dataframe for model training."""
    df = df.copy()
    if "transaction_id" in df.columns:
        df = df.drop(columns=["transaction_id"])

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=["merchant_category", "transaction_type"], drop_first=True)

    # Ensure all columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(0)

    return df


if __name__ == "__main__":
    generate_dataset()
