import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static")
CORS(app)

MODEL_DIR = "models"
model = None
feature_columns = None
model_info = None

# In-memory stores (fallback when MongoDB unavailable)
transactions_store = []
alerts_store = []

# MongoDB (optional)
mongo_client = None
db = None

def init_mongo():
    global mongo_client, db
    mongo_uri = os.environ.get("MONGODB_URI", "")
    if mongo_uri:
        try:
            from pymongo import MongoClient
            mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
            mongo_client.admin.command("ping")
            db = mongo_client["fraud_detection"]
            print("Connected to MongoDB")
        except Exception as e:
            print(f"MongoDB not available, using in-memory storage: {e}")
            db = None
    else:
        print("No MONGODB_URI set, using in-memory storage")

def load_model():
    global model, feature_columns, model_info
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    cols_path = os.path.join(MODEL_DIR, "feature_columns.pkl")
    info_path = os.path.join(MODEL_DIR, "model_info.json")

    if os.path.exists(model_path) and os.path.exists(cols_path):
        model = joblib.load(model_path)
        feature_columns = joblib.load(cols_path)
        print(f"Loaded model with {len(feature_columns)} features")
    else:
        print("WARNING: No trained model found. Run train.py first.")

    if os.path.exists(info_path):
        with open(info_path) as f:
            model_info = json.load(f)
#----------------------------
def store_transaction(txn):
    if db is not None:
        db.transactions.insert_one(txn)
    else:
        transactions_store.append(txn)

def store_alert(alert):
    if db is not None:
        db.alerts.insert_one(alert)
    else:
        alerts_store.append(alert)

def get_transactions(limit=100):
    if db is not None:
        return list(db.transactions.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
    return sorted(transactions_store, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]

def get_alerts(limit=50):
    if db is not None:
        return list(db.alerts.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
    return sorted(alerts_store, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]


# ─── Routes ───

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "mongodb_connected": db is not None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None or feature_columns is None:
        return jsonify({"error": "Model not loaded. Train the model first."}), 503

    data = request.get_json(force=True, silent=True)
    if not data:
        # Fallback: try parsing raw body
        try:
            data = json.loads(request.data)
        except Exception:
            pass
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        df = pd.DataFrame([data])
        txn_id = data.get("transaction_id", f"TXN{datetime.now().strftime('%Y%m%d%H%M%S')}")

        if "transaction_id" in df.columns:
            df = df.drop(columns=["transaction_id"])
        if "is_fraud" in df.columns:
            df = df.drop(columns=["is_fraud"])

        df = pd.get_dummies(df, columns=["merchant_category", "transaction_type"], drop_first=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.fillna(0)

        # Align columns
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]

        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1]) if hasattr(model, "predict_proba") else None

        result = {
            "transaction_id": txn_id,
            "is_fraud": prediction,
            "fraud_probability": round(probability, 4) if probability is not None else None,
            "risk_level": "HIGH" if (probability or 0) > 0.7 else "MEDIUM" if (probability or 0) > 0.4 else "LOW",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Store
        txn_record = {**data, **result}
        store_transaction(txn_record)

        if prediction == 1:
            alert = {
                "transaction_id": txn_id,
                "fraud_probability": result["fraud_probability"],
                "risk_level": result["risk_level"],
                "amount": data.get("amount"),
                "timestamp": result["timestamp"],
                "status": "open",
            }
            store_alert(alert)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/transactions", methods=["GET"])
def list_transactions():
    limit = request.args.get("limit", 100, type=int)
    return jsonify(get_transactions(limit))


@app.route("/api/alerts", methods=["GET"])
def list_alerts():
    limit = request.args.get("limit", 50, type=int)
    return jsonify(get_alerts(limit))


@app.route("/api/alerts/<alert_id>/resolve", methods=["POST"])
def resolve_alert(alert_id):
    if db is not None:
        db.alerts.update_one({"transaction_id": alert_id}, {"$set": {"status": "resolved"}})
    else:
        for a in alerts_store:
            if a["transaction_id"] == alert_id:
                a["status"] = "resolved"
                break
    return jsonify({"status": "resolved", "transaction_id": alert_id})


@app.route("/api/model/info", methods=["GET"])
def get_model_info():
    if model_info is None:
        return jsonify({"error": "No model info available"}), 404
    return jsonify(model_info)


@app.route("/api/model/retrain", methods=["POST"])
def retrain():
    try:
        from train import train_models
        best_name, results = train_models()
        load_model()
        return jsonify({"message": "Retrained successfully", "best_model": best_name, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Generate and predict a batch of random transactions for demo."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    count = request.json.get("count", 10) if request.json else 10
    count = min(count, 50)

    results = []
    for i in range(count):
        is_fraud_like = np.random.random() < 0.15
        if is_fraud_like:
            txn = {
                "amount": round(float(np.random.lognormal(6, 1.5)), 2),
                "hour_of_day": int(np.random.choice([0, 1, 2, 3, 23])),
                "day_of_week": int(np.random.randint(0, 7)),
                "merchant_category": np.random.choice(["electronics", "travel", "online"]),
                "transaction_type": "online",
                "distance_from_home": round(float(np.abs(np.random.normal(200, 150))), 1),
                "distance_from_last_txn": round(float(np.abs(np.random.normal(300, 200))), 1),
                "avg_daily_transactions": float(np.random.poisson(8)),
                "velocity_last_hour": float(np.random.poisson(5)),
                "failed_attempts_24h": int(np.random.choice([2, 3, 4])),
                "account_age_days": int(np.random.randint(1, 30)),
                "is_international": 1,
            }
        else:
            txn = {
                "amount": round(float(np.random.lognormal(4, 1)), 2),
                "hour_of_day": int(np.random.randint(8, 21)),
                "day_of_week": int(np.random.randint(0, 7)),
                "merchant_category": np.random.choice(["grocery", "restaurant", "gas", "clothing"]),
                "transaction_type": np.random.choice(["card_present", "contactless"]),
                "distance_from_home": round(float(np.abs(np.random.normal(10, 15))), 1),
                "distance_from_last_txn": round(float(np.abs(np.random.normal(5, 10))), 1),
                "avg_daily_transactions": float(np.random.poisson(3)),
                "velocity_last_hour": float(np.random.poisson(1)),
                "failed_attempts_24h": 0,
                "account_age_days": int(np.random.randint(100, 3000)),
                "is_international": 0,
            }

        # Predict via internal logic
        txn["transaction_id"] = f"SIM{datetime.now().strftime('%H%M%S')}{i:03d}"
        df = pd.DataFrame([txn])
        df_pred = df.drop(columns=["transaction_id"])
        df_pred = pd.get_dummies(df_pred, columns=["merchant_category", "transaction_type"], drop_first=True)
        for col in df_pred.columns:
            df_pred[col] = pd.to_numeric(df_pred[col], errors="coerce")
        df_pred = df_pred.fillna(0)
        for col in feature_columns:
            if col not in df_pred.columns:
                df_pred[col] = 0
        df_pred = df_pred[feature_columns]

        prediction = int(model.predict(df_pred)[0])
        probability = float(model.predict_proba(df_pred)[0][1]) if hasattr(model, "predict_proba") else 0

        result = {
            **txn,
            "is_fraud": prediction,
            "fraud_probability": round(probability, 4),
            "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.4 else "LOW",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        store_transaction(result)
        if prediction == 1:
            store_alert({
                "transaction_id": result["transaction_id"],
                "fraud_probability": result["fraud_probability"],
                "risk_level": result["risk_level"],
                "amount": result["amount"],
                "timestamp": result["timestamp"],
                "status": "open",
            })
        results.append(result)

    fraud_count = sum(1 for r in results if r["is_fraud"] == 1)
    return jsonify({"transactions": results, "total": len(results), "fraud_detected": fraud_count})


@app.route("/api/stats", methods=["GET"])
def stats():
    txns = get_transactions(1000)
    total = len(txns)
    fraud = sum(1 for t in txns if t.get("is_fraud") == 1)
    legit = total - fraud

    alerts = get_alerts(1000)
    open_alerts = sum(1 for a in alerts if a.get("status") == "open")
    resolved_alerts = sum(1 for a in alerts if a.get("status") == "resolved")

    amounts = [t.get("amount", 0) for t in txns if t.get("is_fraud") == 1]
    avg_fraud_amount = round(sum(amounts) / len(amounts), 2) if amounts else 0

    return jsonify({
        "total_transactions": total,
        "fraud_transactions": fraud,
        "legit_transactions": legit,
        "fraud_rate": round(fraud / total * 100, 2) if total > 0 else 0,
        "open_alerts": open_alerts,
        "resolved_alerts": resolved_alerts,
        "avg_fraud_amount": avg_fraud_amount,
    })


init_mongo()
load_model()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
