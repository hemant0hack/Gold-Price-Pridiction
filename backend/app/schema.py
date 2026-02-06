from pydantic import BaseModel

class PredictionResponse(BaseModel):
    symbol: str
    last_close: float
    predicted_price: float
    trend: str
