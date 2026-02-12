# AI Advanced Fraud Detection System (AFDS)

Real-time fraud detection system using Machine Learning to analyze transaction data and identify fraudulent activities in banking, insurance, and e-commerce.

## Tech Stack

- **Python** - Backend & ML
- **Flask** - REST API framework
- **Scikit-learn & XGBoost** - ML model training
- **MongoDB** - Transaction & alert storage (optional, falls back to in-memory)
- **Pandas & NumPy** - Data preprocessing
- **Gunicorn** - Production WSGI server

## How It Works

1. **Data Collection** - Transactional and user data is collected and stored
2. **Data Preprocessing** - Cleaning, encoding, and normalization applied
3. **Model Training** - ML algorithms (Logistic Regression, Random Forest, XGBoost) trained to classify fraud vs. legitimate transactions
4. **Model Prediction** - Trained model predicts whether a new transaction is fraudulent
5. **Alerts & Monitoring** - Flags and alerts administrators for suspicious transactions

## Features

- 3 ML models compared, best auto-selected by F1 score
- Real-time fraud probability with risk levels (LOW / MEDIUM / HIGH)
- Interactive dashboard with stats, transaction list, alerts, and simulation
- On-demand model retraining via API
- MongoDB integration (optional)

## Project Structure

```
├── app.py              # Flask backend with REST APIs
├── train.py            # ML model training pipeline
├── data_utils.py       # Synthetic data generation & preprocessing
├── test_api.py         # API test script
├── static/
│   ├── index.html      # Dashboard UI
│   ├── style.css       # Dashboard styles
│   └── script.js       # Dashboard logic
├── models/             # Saved trained models (generated)
├── Procfile            # Railway deployment config
├── railway.json        # Railway builder settings
├── requirements.txt    # Python dependencies
└── runtime.txt         # Python version
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/predict` | Classify a transaction as fraud/legit |
| `POST` | `/api/simulate` | Generate random transactions for demo |
| `GET` | `/api/transactions` | List all transactions |
| `GET` | `/api/alerts` | List fraud alerts |
| `POST` | `/api/alerts/<id>/resolve` | Resolve an alert |
| `GET` | `/api/stats` | Dashboard statistics |
| `GET` | `/api/model/info` | Model performance metrics |
| `POST` | `/api/model/retrain` | Retrain models on demand |
| `GET` | `/api/health` | Health check |

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python train.py

# Start development server
python app.py
```

The dashboard will be available at `http://localhost:5000`.

## Prediction Request Example

```json
POST /api/predict
{
  "amount": 5000,
  "hour_of_day": 2,
  "day_of_week": 5,
  "merchant_category": "electronics",
  "transaction_type": "online",
  "distance_from_home": 500,
  "is_international": 1,
  "velocity_last_hour": 10,
  "account_age_days": 30,
  "avg_daily_transactions": 2
}
```

## Deployment (Railway)

1. Push code to a GitHub repository
2. Go to [railway.app](https://railway.app) and sign in with GitHub
3. Click **New Project** > **Deploy from GitHub Repo**
4. Select the repository — Railway auto-detects the config
5. (Optional) Add `MONGODB_URI` in the Variables tab for persistent storage
6. Generate a public domain under **Settings > Networking**

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MONGODB_URI` | MongoDB connection string | No (uses in-memory storage if absent) |
| `PORT` | Server port (set by Railway automatically) | No (defaults to 5000) |
