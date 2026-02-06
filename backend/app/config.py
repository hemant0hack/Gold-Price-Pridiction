import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/stock_model.pkl")
DATA_PERIOD = os.getenv("DATA_PERIOD", "2y")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
