# app.py
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import onnxruntime as ort
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

# Load models
anomaly_model = joblib.load("../Models/xgboost_12318.pkl")

# ONNX model session (for CNN-LSTM RUL prediction)
cnn_lstm_session = ort.InferenceSession("../Models/cnn_lstm_pm_enhanced.onnx")


# API Key for basic authentication
API_KEY = "key123456"

# InfluxDB settings
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "THptiWTEs5bZCQudb8qDcrfBDTyXrbuF_vEttYM1baTsUicwfYa7fbgpRFJUQIT6rDoUwH1puHnU-pC20vYwSg=="
INFLUX_ORG = "Generator"
INFLUX_BUCKET = "RT_Data"

# Data model
class DataPoint(BaseModel):
    air_id: str
    features: dict

# Utility: fetch InfluxDB data
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

# Utility: align features for anomaly model
def align_features_to_model(df_features: pd.DataFrame, model) -> pd.DataFrame:
    model_cols = model.get_booster().feature_names
    for col in model_cols:
        if col not in df_features.columns:
            df_features[col] = 0
    return df_features[model_cols]

# endpoints 

@app.get("/test_db_connection")
def test_db_connection(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        df = client.query_api().query_data_frame(
            f'from(bucket:"{INFLUX_BUCKET}") |> range(start: -1h) '
        )
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
        return {"message": f"No data found for AIR {air_id}"}
    return df_raw.to_dict(orient="records")

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

# for cnn ltsm model 
# load scaler used in training
rul_scaler = joblib.load("../Models/scaler_cnn_lstm_pm_enhanced.pkl")

# define exact feature order used in training
RUL_FEATURE_ORDER = [
    'va_V','vb_V','vc_V','va-vb_V','vb-vc_V','vc-va_V',
    'ia_A','ib_A','ic_A',
    'ptot_W','qtot_Var','stot_VA','pa_W','pb_W','pc_W',
    'pfa_None','pfb_None','pfc_None','pftot_None',
    'temp_Degrees Celsius','pressure_Bar','fuel_%','freq_Hz*10',
    'current_imbalance','voltage_imbalance','pf_anomaly',
    'temp_Degrees Celsius_roc','fuel_%_roc',
    'ptot_W_rollmean','ptot_W_rollstd','ia_A_rollmean','ia_A_rollstd','pf_anomaly_rollmean','pf_anomaly_rollstd',
    'is_running'
]

@app.get("/predict_cnn_lstm")
def predict_cnn_lstm(air_id: str, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if cnn_lstm_session is None:
        raise HTTPException(status_code=500, detail="ONNX CNN-LSTM model not loaded")

    df_raw = fetch_influx_data(air_id)
    if df_raw.empty:
        return {"rul": None, "message": f"No new data for AIR {air_id}"}

    try:
        numeric_df = df_raw.select_dtypes(include='number')
        df_features = feature_engineering_dispatcher(numeric_df, air_id)

        # Ensure all features exist
        for f in RUL_FEATURE_ORDER:
            if f not in df_features.columns:
                df_features[f] = 0

        # Align columns
        df_aligned = df_features[RUL_FEATURE_ORDER]

        # Pad or truncate to 60 timesteps
        seq_len = 60
        if len(df_aligned) < seq_len:
            pad = pd.DataFrame(0, index=range(seq_len - len(df_aligned)), columns=df_aligned.columns)
            df_aligned = pd.concat([pad, df_aligned], ignore_index=True)
        elif len(df_aligned) > seq_len:
            df_aligned = df_aligned.tail(seq_len).reset_index(drop=True)

        # Scale features
        X_scaled = rul_scaler.transform(df_aligned.values.astype(np.float32))

        # Add batch dimension for ONNX input
        X_input = np.expand_dims(X_scaled, axis=0)

        # Get input/output names
        input_name = cnn_lstm_session.get_inputs()[0].name
        output_name = cnn_lstm_session.get_outputs()[0].name

        preds = cnn_lstm_session.run([output_name], {input_name: X_input})[0][0]

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
        raise HTTPException(status_code=500, detail=f"CNN-LSTM prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "anomaly_model_loaded": anomaly_model is not None,
        "cnn_lstm_loaded": cnn_lstm_session is not None
    }
