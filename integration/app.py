# app.py
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import joblib
import pandas as pd
from influxdb_client import InfluxDBClient
from feature_engineering import feature_engineering_dispatcher

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
model= joblib.load("../Models/xgboost_12318.pkl")

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

# helper function to fetch raw data from Influx 
def fetch_influx_data(air_id: str, start="-1m") -> pd.DataFrame:
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    query_api = client.query_api()

    query = f'''
        from(bucket: "RT_Data")
        |> range(start: -1h)   // last hour, adjust as needed
        |> filter(fn: (r) => r._measurement == "Epi")
        |> pivot(
            rowKey: ["_time"],
            columnKey: ["_field"],
            valueColumn: "_value"
        )
    '''
    df = query_api.query_data_frame(query)
    df['time'] = pd.to_datetime(df['_time'], errors='coerce')
    df = df.sort_values('time').reset_index(drop=True)
    return df

# prediction endpoint using latest Influx data 
@app.get("/predict_latest")
def predict_latest(air_id: str, x_api_key: str = Header(None)):
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
    df_features = feature_engineering_dispatcher(df_raw, data.air_id)

    pred = model.predict(df_features)[0]
    prob = model.predict_proba(df_features)[0,1]

    return {"anomaly": int(pred), "probability": float(prob), "air_id": data.air_id}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}
