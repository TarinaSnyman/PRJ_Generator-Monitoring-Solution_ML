# app.py
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import joblib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from influxdb_client import InfluxDBClient
from feature_engineering import feature_engineering_dispatcher

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
model= joblib.load("../Models/xgboost_12318.plk")

# API Key for basic authentication 
API_KEY = "key123456" # hardcoded for now

# InfluxDB settings 
INFLUX_URL = "http://your-influxdb:8086"
INFLUX_TOKEN = "YTHptiWTEs5bZCQudb8qDcrfBDTyXrbuF_vEttYM1baTsUicwfYa7fbgpRFJUQIT6rDoUwH1puHnU-pC20vYwSg=="
INFLUX_ORG = "Generator"
INFLUX_BUCKET = "RT_Data"

# data model for optional POST requests 
class DataPoint(BaseModel):
    air_id: str
    features: dict

def fetch_influx_data(air_id: str, start="-1h") -> pd.DataFrame:
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    query_api = client.query_api()
    query = f'''
        from(bucket: "{INFLUX_BUCKET}")
        |> range(start: {start})
        |> filter(fn: (r) => r._measurement == "{air_id}")
        |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df = query_api.query_data_frame(query)
    client.close()
    
    if "_time" in df.columns:
        df['time'] = pd.to_datetime(df['_time'], errors='coerce')
        df = df.sort_values('time').reset_index(drop=True)
        df = df.drop(columns=["_time"], errors="ignore")
    
    return df

def align_features_to_model(df_features: pd.DataFrame, model) -> pd.DataFrame:
    # Fill missing columns with 0
    model_cols = model.get_booster().feature_names
    for col in model_cols:
        if col not in df_features.columns:
            df_features[col] = 0
    # Keep only model columns and preserve order
    return df_features[model_cols]

# Endpoints
# test db conn
@app.get("/test_db_connection")
def test_db_connection(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        df = client.query_api().query_data_frame(f'from(bucket:"{INFLUX_BUCKET}") |> range(start: -1m) |> limit(n:1)')
        client.close()
        return {"status": "ok", "rows_returned": len(df)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/fetch_data")
def fetch_data(air_id: str, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    df_raw = fetch_influx_data(air_id)
    if df_raw.empty:
        return {"anomaly": None, "message": "No new data for AIR " + air_id}

    # Feature engineering
    df_features = feature_engineering_dispatcher(df_raw, air_id)

    # Prediction (last row for real-time)
    pred = model.predict(df_features)[-1]
    prob = model.predict_proba(df_features)[-1,1]

    return {"anomaly": int(pred), "probability": float(prob), "air_id": air_id}

# Optional endpoint: predict from POSTed raw features
@app.post("/predict")
def predict_post(data: DataPoint, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    df_raw = pd.DataFrame([data.features])
    try:
        df_features = feature_engineering_dispatcher(df_raw, data.air_id)
        pred = anomaly_model.predict(df_features)[0]
        prob = anomaly_model.predict_proba(df_features)[0, 1]
        return {"anomaly": int(pred), "probability": float(prob), "air_id": data.air_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"POST prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok", "anomaly_model_loaded": anomaly_model is not None}
