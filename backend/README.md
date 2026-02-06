Backend for Gold-Silver Price Prediction

Run locally (Windows PowerShell):

```powershell
cd backend
venv\Scripts\activate
python -m uvicorn app.main:app --reload --port 8000
```

Endpoints:
- GET / -> status
- GET /health -> health
- POST /train/{symbol} -> train model
- GET /predict/{symbol} -> predict
