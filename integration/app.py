# app.py
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import onnxruntime as ort
from influxdb_client import InfluxDBClient
from feature_engineering import (feature_engineering_dispatcher, 
                                 feature_engineering_dispatcher_cnnlstm,
                                 preprocess_for_cnn_lstm)
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
# anomaly detection with xgboost
anomaly_model_12318 = joblib.load("../Models/xgboost_12318.pkl")
anomaly_model_12300 = joblib.load("../Models/xgboost_12300.pkl")
anomaly_model_12305 = joblib.load("../Models/xgboost_12305.pkl")

# CNN-LSTM prediction
cnn_lstm_model_12300 = ort.InferenceSession("../Models/cnn_lstm_pm_12300_optimized.onnx")
cnn_lstm_model_12305 = ort.InferenceSession("../Models/cnn_lstm_pm_12305.onnx")
cnn_lstm_model_12318 = ort.InferenceSession("../Models/cnn_lstm_pm_12318_optimized.onnx")

scaler_12300=joblib.load("../Models/scaler_info_cnn_lstm_pm_12300_optimized.pkl")
scaler_12305=joblib.load("../Models/scaler_info_cnn_lstm_pm_12305.pkl")
scaler_12318=joblib.load("../Models/scaler_info_cnn_lstm_pm_12318_optimized.pkl")

# Load feature info
features_12300 = joblib.load("../Models/feature_info_cnn_lstm_pm_12300_optimized.pkl")
features_12305 = joblib.load("../Models/feature_info_cnn_lstm_pm_12305.pkl")
features_12318 = joblib.load("../Models/feature_info_cnn_lstm_pm_12318_optimized.pkl")


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


# utility:
# fetch InfluxDB data
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

# align features for anomaly model
def align_features_to_model(df_features: pd.DataFrame, model) -> pd.DataFrame:
    model_cols = model.get_booster().feature_names
    for col in model_cols:
        if col not in df_features.columns:
            df_features[col] = 0
    return df_features[model_cols]

# endpoints:
# test the connection to the database
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
    
# get data from the database --> why only returning 30 rows?
@app.get("/fetch_data")
def fetch_data(air_id: str, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    df_raw = fetch_influx_data(air_id)
    if df_raw.empty:
        return {"message": f"No data found for AIR {air_id}"}
    return df_raw.to_dict(orient="records")
# gets feature engineared data -->just used for testing purposes 
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
    
#model endpoints: --> still need to add models for dataset 12300 and 21305
# realtimde data from the database
# xgboost models
@app.get("/predict_anomaly")
def predict_anomaly(air_id: str, source: str = "influx", x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    debug_info = {}  # will collect debug info to return

    try:
        # load data
        if source == "on":
            csv_path = f"../data/simulatedData/onData/air{air_id}_SimulatedData.csv"
            debug_info['csv_path'] = csv_path
            df_raw = pd.read_csv(csv_path)
        elif source == "failing":
            csv_path = f"../data/simulatedData/onFailingData/air{air_id}_SimulatedFailingData.csv"
            debug_info['csv_path'] = csv_path
            df_raw = pd.read_csv(csv_path)
        elif source == "influx":
            debug_info['source'] = "influx"
            df_raw = fetch_influx_data(air_id)
        else:
            raise HTTPException(status_code=400, detail="Invalid source type. Use 'influx' or 'csv'.")

        debug_info['raw_shape'] = df_raw.shape
        debug_info['raw_columns'] = df_raw.columns.tolist()

        if df_raw.empty:
            return {"anomaly": None, "message": f"No new data for AIR {air_id}", "debug": debug_info}

        # feature engineering
        numeric_df = df_raw.select_dtypes(include='number')
        debug_info['numeric_columns'] = numeric_df.columns.tolist()

        try:
            df_features = feature_engineering_dispatcher(numeric_df, air_id)
            debug_info['features_shape'] = df_features.shape
            debug_info['features_columns'] = df_features.columns.tolist()
        except Exception as fe_err:
            debug_info['feature_engineering_error'] = str(fe_err)
            return {"error": "Feature engineering failed", "debug": debug_info}

        # select model
        if air_id == "12318" or air_id == "Epi":
            model = anomaly_model_12318
        elif air_id == "12300" or air_id == "Military1":
            model = anomaly_model_12300
        elif air_id == "12305" or air_id == "Military2":
            model = anomaly_model_12305
        else:
            debug_info['model_error'] = f"No model available for AIR {air_id}"
            return {"error": "Model selection failed", "debug": debug_info}

        #Align features
        df_features = align_features_to_model(df_features, model)
        debug_info['aligned_columns'] = df_features.columns.tolist()

        #predict
        pred = model.predict(df_features)[-1]
        prob = model.predict_proba(df_features)[-1, 1]

        return {
            "air_id": air_id,
            "source": source,
            "anomaly": int(pred),
            "probability": float(prob),
            "debug": debug_info
        }

    except Exception as e:
        import traceback
        debug_info['exception'] = str(e)
        debug_info['traceback'] = traceback.format_exc()
        return {"error": "Anomaly prediction failed", "debug": debug_info}


# # cnn ltsm model 

@app.get("/predict_failure")
def predict_failure(air_id: str, source: str = "influx", x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    debug_info = {}
    
    try:
        # load data
        if source == "csv":
            csv_path = f"../data/csvData/air{air_id}.csv"
            debug_info['csv_path'] = csv_path
            df_raw = pd.read_csv(csv_path)
        elif source == "influx":
            df_raw = fetch_influx_data(air_id)
            debug_info['source'] = "influx"
        else:
            raise HTTPException(status_code=400, detail="Invalid source type")

        if df_raw.empty:
            return {"air_id": air_id, "message": "No data available", "debug": debug_info}
        debug_info["raw_shape"] = df_raw.shape

        # feature engineering (CNN-LSTM version)
        numeric_df = df_raw.select_dtypes(include="number")
        try:
            df_features = feature_engineering_dispatcher_cnnlstm(numeric_df, air_id)
            debug_info["features_shape"] = df_features.shape
        except Exception as fe_err:
            debug_info["feature_engineering_error"] = str(fe_err)
            return {"error": "Feature engineering failed", "debug": debug_info}

        # Select model and scaler
        if air_id == "12318":
            model = cnn_lstm_model_12318
            scaler = scaler_12318
            feature_info = features_12318
        elif air_id == "12300":
            model = cnn_lstm_model_12300
            scaler = scaler_12300
            feature_info = features_12300
        elif air_id == "12305":
            model = cnn_lstm_model_12305
            scaler = scaler_12305
            feature_info = features_12305
        else:
            raise HTTPException(status_code=400, detail=f"No model for AIR {air_id}")

        #Preprocessing
        X = preprocess_for_cnn_lstm(df_features, scaler, feature_info, timesteps=60)
        if X.shape[0] == 0:
            raise HTTPException(status_code=400, detail="Not enough data for CNN-LSTM sequence generation")

        debug_info["input_shape"] = X.shape

        #run ONNX model
        input_name = model.get_inputs()[0].name
        preds = model.run(None, {input_name: X.astype(np.float32)})[0][-1]

        horizons = [1, 2, 4, 6]
        probabilities = {f"{h}h": float(preds[i]) for i, h in enumerate(horizons)}

        return {
            "air_id": air_id,
            "source": source,
            "probabilities": probabilities,
            "debug": debug_info
        }

    except Exception as e:
        import traceback
        debug_info["exception"] = str(e)
        debug_info["traceback"] = traceback.format_exc()
        return {"error": "Prediction failed", "debug": debug_info}



# checks the status and that the models exist
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "anomaly_model_12318_loaded": anomaly_model_12318 is not None,
        "anomaly_model_12305_loaded": anomaly_model_12305 is not None,
        # "cnn_lstm_loaded": cnn_lstm_session is not None
    }
