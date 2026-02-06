from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schema import PredictionResponse
from .data import fetch_stock_data
from .features import add_features
from .model import train_and_save_model, load_model, predict

app = FastAPI(title="Gold-Silver Price Prediction API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Backend running successfully"}

@app.post("/train/{symbol}")
def train_model(symbol: str):
    """Train model for a given symbol"""
    try:
        df = fetch_stock_data(symbol)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for symbol: {symbol}")
        df = add_features(df)
        model = train_and_save_model(df)
        return {"message": f"Model trained successfully for {symbol}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/predict/{symbol}", response_model=PredictionResponse)
def get_prediction(symbol: str):
    """Get price prediction for a symbol"""
    try:
        df = fetch_stock_data(symbol)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for symbol: {symbol}")
        df = add_features(df)
        model = load_model()
        
        last_row = df.iloc[-1]
        predicted_price = predict(model, last_row)
        last_close = float(last_row["Close"])
        
        trend = "UP" if predicted_price > last_close else "DOWN"
        
        return PredictionResponse(
            symbol=symbol,
            last_close=last_close,
            predicted_price=predicted_price,
            trend=trend
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

