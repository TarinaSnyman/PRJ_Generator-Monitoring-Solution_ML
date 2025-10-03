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

# load models
anomaly_model = joblib.load("../Models/xgboost_12318.pkl")
cnn_lstm_model = load_model("../Models/cnn_lstm_model.h5")

API_KEY = "key123456"
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = 'THptiWTEs5bZCQudb8qDcrfBDTyXrbuF_vEttYM1baTsUicwfYa7fbgpRFJUQIT6rDoUwH1puHnU-pC20vYwSg=='
INFLUX_ORG = "Generator"
INFLUX_BUCKET = "RT_Data"

# helpr functions
class DataPoint(BaseModel):
    air_id: str
    features: dict  # raw sensor values if sending directly

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
        return {"message": f"No data found for AIR {air_id}"}
    return df_raw.to_dict(orient="records")


# Test if feature engineering is done
@app.get("/test_features")
def test_features(air_id: str, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    df_raw = fetch_influx_data(air_id)
    if df_raw.empty:
        return {"message": f"No data found for AIR {air_id}"}
    try:
        numeric_df = df_raw.select_dtypes(include='number')
        df_features = feature_engineering_dispatcher(numeric_df, air_id)
        return df_features.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature engineering error: {str(e)}")


# test xgboost anomaly detection model
@app.get("/predict_anomaly")
def predict_anomaly(air_id: str, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    df_raw = fetch_influx_data(air_id)
    if df_raw.empty:
        return {"anomaly": None, "message": f"No new data for AIR {air_id}"}
    try:
        numeric_df = df_raw.select_dtypes(include='number')
        df_features = feature_engineering_dispatcher(numeric_df, air_id)
        df_features = align_features_to_model(df_features, anomaly_model)
        pred = anomaly_model.predict(df_features)[-1]
        prob = anomaly_model.predict_proba(df_features)[-1, 1]
        return {"anomaly": int(pred), "probability": float(prob), "air_id": air_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly prediction error: {str(e)}")

# ltsm-cnn model
@app.get("/predict_rul")
def predict_rul(air_id: str, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if cnn_lstm_model is None:
        raise HTTPException(status_code=500, detail="CNN-LSTM model not loaded")

    df_raw = fetch_influx_data(air_id)
    if df_raw.empty:
        return {"rul": None, "message": f"No new data for AIR {air_id}"}
    
    try:
        numeric_df = df_raw.select_dtypes(include='number')
        df_features = feature_engineering_dispatcher(numeric_df, air_id)

        # Adapt input shape: (1, timesteps, features)
        X_input = np.expand_dims(df_features.values, axis=0)
        preds = cnn_lstm_model.predict(X_input)[0]  # [p_1h, p_2h, p_4h, p_6h]

        return {
            "air_id": air_id,
            "failure_probabilities": {
                "1_hour": float(preds[0]),
                "2_hours": float(preds[1]),
                "4_hours": float(preds[2]),
                "6_hours": float(preds[3]),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RUL prediction error: {str(e)}")


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
