from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os
from .config import MODEL_PATH

def train_and_save_model(df):
    """Train and save the model"""
    X = df[["MA20", "MA50", "RSI"]]
    y = df["Close"]

    X_train, _, y_train, _ = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model

def load_model():
    """Load model with error handling"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
    return joblib.load(MODEL_PATH)

def predict(model, last_row):
    """Make prediction"""
    X = [[last_row["MA20"], last_row["MA50"], last_row["RSI"]]]
    return float(model.predict(X)[0])
