---
title: House Price Prediction API
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🏠 House Price Prediction API

An end-to-end ML API that predicts US residential house sale prices using **XGBoost**, deployed on Hugging Face Spaces.

## Stack
- **Model**: XGBoost (tuned with Optuna, 89 features)
- **API**: FastAPI + Uvicorn
- **Cloud**: Hugging Face Spaces (Docker)

## Usage

Send a POST request to `/predict` with house features:

```json
{
  "OverallQual": 7,
  "GrLivArea": 1710,
  "GarageCars": 2,
  "TotalBsmtSF": 856,
  "FirstFlrSF": 856,
  "SecondFlrSF": 854,
  "FullBath": 2,
  "YearBuilt": 2003,
  "YearRemodAdd": 2003,
  "YrSold": 2010,
  "MoSold": 2,
  "Neighborhood": "CollgCr",
  "MSZoning": "RL",
  "LotArea": 8450
}
```

Response:
```json
{
  "predicted_price_usd": 205414.0,
  "predicted_price_formatted": "$205,414"
}
```

## Endpoints
- `GET /` — API info
- `GET /health` — health check
- `POST /predict` — get price prediction
- `GET /docs` — interactive Swagger UI
