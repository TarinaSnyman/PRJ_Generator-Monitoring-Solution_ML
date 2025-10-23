# feature_engineering.py
import pandas as pd
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture

# helper function to map raw influx columns to model features 
def map_influx_to_model_columns_12318(df: pd.DataFrame) -> pd.DataFrame:
    column_mapping = {
        'ia':'ia_A', 'ib':'ib_A', 'ic':'ic_A',
        'ptot':'ptot_W', 'pftot':'pftot_None',
        'va':'va_V', 'vb':'vb_V', 'vc':'vc_V',
        'va-vb':'va-vb_V', 'vb-vc':'vb-vc_V', 'vc-va':'vc-va_V',
        'pa':'pa_W', 'pb':'pb_W', 'pc':'pc_W',
        'qa':'qa_Var', 'qb':'qb_Var', 'qc':'qc_Var',
        'sa':'sa_VA', 'sb':'sb_VA', 'sc':'sc_VA',
        'expwh':'expwh_Kwh*10', 'expvar':'expvar_Kvarh*10',
        'freq':'freq_Hz*10', 'temp':'temp_Degrees Celsius',
        'pressure':'pressure_Bar', 'fuel':'fuel_%', 'vbat':'vbat_V', 'hours':'hours_sec'
    }
    df = df.rename(columns=column_mapping)
    return df

def map_influx_to_model_columns_123005(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "result": "id",
        "table": "epoch",
        "_start": "start_time",
        "_stop": "stop_time",
        "_time": "time",
        "_measurement": "air",
        "app": "app_none",
        "device_id": "device",
        "cooltemp": "cooltemp_degree-celsius",
        "expvar": "expvar_var-hour",
        "expwh": "expwh_watt-hour",
        "freq": "freq_hertz",
        "hours": "hours_hour",
        "ia": "ia_ampere",
        "iavg": "iavg_ampere",
        "ib": "ib_ampere",
        "ic": "ic_ampere",
        "oilpress": "oilpress_pascal",
        "pftot": "pftot_none",
        "ptot": "ptot_watt",
        "ptotper": "ptotper_percent",
        "qtot": "qtot_var",
        "qtotper": "qtotper_percent",
        "rpm": "rpm_revolutions-per-minute",
        "servday": "servday_day",
        "servhr": "servhr_hour",
        "startcount": "startcount_none",
        "state": "state_none",
        "stot": "stot_volt-ampere",
        "stotper": "stotper_percent",
        "va": "va_volt",
        "vb": "vb_volt",
        "vc": "vc_volt",
        "vlineavg": "vlineavg_volt",
        "vlineper": "vlineper_volt",
        "vbat": "vbat_volt"
    }

    # Apply mapping only for existing columns to avoid KeyErrors
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    # drop unneeded columns
    drop_cols = ["result", "table", "_start", "_stop"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df


def feature_engineering_air12318(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ON/OFF detection
    if "ptot_W" in df.columns:
        p = df["ptot_W"].clip(lower=0).fillna(0)
        X = np.log1p(p).values.reshape(-1, 1)
        try:
            gmm = GaussianMixture(n_components=2, random_state=42).fit(X)
            labels = gmm.predict(X)
            means = gmm.means_.flatten()
            on_cluster = np.argmax(means)
            df["is_running_gmm"] = (labels == on_cluster).astype(int)
        except Exception:
            df["is_running_gmm"] = 0
    else:
        df["is_running_gmm"] = 0

    # any current present
    current_cols = [c for c in ["ia_A", "ib_A", "ic_A"] if c in df.columns]
    if current_cols:
        df["any_current"] = df[current_cols].sum(axis=1) > 0.1
    else:
        df["any_current"] = 0

    # final is_running flag
    df["is_running"] = ((df["is_running_gmm"] == 1) | (df["any_current"] == 1)).astype(int)


# Feature engineering
    # Current imbalance
    if len(current_cols) == 3:
        df["current_imbalance"] = df[current_cols].std(axis=1)
    else:
        df["current_imbalance"] = 0

    # PF anomaly
    if "pftot_None" in df.columns:
        df["pf_anomaly"] = np.abs(1 - df["pftot_None"])
    else:
        df["pf_anomaly"] = 0

    # Temperature rate-of-change
    temp_cols = [c for c in df.columns if "temp" in c.lower()]
    for col in temp_cols:
        df[f"{col}_roc"] = df[col].diff().fillna(0)

    # Fuel rate-of-change
    fuel_cols = [c for c in df.columns if "fuel" in c.lower()]
    for col in fuel_cols:
        df[f"{col}_roc"] = df[col].diff().fillna(0)

    # Rolling statistics (use is_running mask so it only applies while running)
    rolling_window = 60
    for col in ["ptot_W", "ia_A", "pf_anomaly"]:
        if col in df.columns:
            df[f"{col}_rollmean"] = (
                df[col].where(df["is_running"] == 1)
                .rolling(window=rolling_window, min_periods=1)
                .mean()
                .fillna(0)
            )
            df[f"{col}_rollstd"] = (
                df[col].where(df["is_running"] == 1)
                .rolling(window=rolling_window, min_periods=1)
                .std()
                .fillna(0)
            )
        else:
            df[f"{col}_rollmean"] = 0
            df[f"{col}_rollstd"] = 0
    # Cleanup & feature selection
    exclude_cols = ["time", "iforest_anomaly", "heuristic_anomaly"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    return df[feature_cols]



# feature engineering for 12305
def feature_engineering_air12305(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Detect ON/OFF using GMM
    if 'ptot_watt' in df.columns:
        gmm = GaussianMixture(n_components=2, random_state=42)
        labels = gmm.fit_predict(df[['ptot_watt']].fillna(0))
        running_label = np.argmax(gmm.means_.flatten())
        df['is_running'] = (labels == running_label).astype(int)
    else:
        df['is_running'] = 0

    # PF anomaly
    if 'pftot_none' in df.columns:
        df['pf_anomaly'] = np.abs(1 - df['pftot_none'].clip(0, 1))
    else:
        df['pf_anomaly'] = np.nan

    # Current imbalance
    if all(c in df.columns for c in ['ia_ampere', 'ib_ampere', 'ic_ampere']):
        df['current_imbalance'] = df[['ia_ampere','ib_ampere','ic_ampere']].std(axis=1) / \
                                  df[['ia_ampere','ib_ampere','ic_ampere']].mean(axis=1)
    else:
        df['current_imbalance'] = 0

    # Power ratio
    if all(c in df.columns for c in ['ptot_watt','stot_volt-ampere']):
        df['power_ratio'] = df['ptot_watt'] / df['stot_volt-ampere'].replace(0, np.nan)
    else:
        df['power_ratio'] = 0

    # Rolling stats
    ROLL_WINDOW = 10
    if 'cooltemp_degree-celsius' in df.columns:
        df['temp_roll_mean'] = df['cooltemp_degree-celsius'].rolling(ROLL_WINDOW).mean()
        df['temp_roll_std'] = df['cooltemp_degree-celsius'].rolling(ROLL_WINDOW).std()

    if 'ptot_watt' in df.columns:
        df['ptot_watt_rollmean'] = df['ptot_watt'].rolling(ROLL_WINDOW).mean()
        df['ptot_watt_rollstd'] = df['ptot_watt'].rolling(ROLL_WINDOW).std()

    df = df.fillna(0)
    return df


#dispatches the feture engineered dataset for each air
def feature_engineering_dispatcher(df: pd.DataFrame, air_id: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"No data provided for AIR {air_id}")
    
    df_mapped=[]
    if air_id == "12318" or air_id=="Epi":
        # map Influx columns to model features
        df_mapped = map_influx_to_model_columns_12318(df)
        return feature_engineering_air12318(df_mapped)
    elif air_id in ["12300", "12305", "Military1", "Military2"]:
        df_mapped = map_influx_to_model_columns_123005(df)
        return feature_engineering_air12305(df_mapped)
    else:
        raise ValueError(f"Feature engineering not defined for AIR {air_id}")
    
# CNN LTSM
# load scaler once globally

def preprocess_for_cnn_lstm(df, scaler, feature_info, timesteps=60):
    # Align columns
    df = df[feature_info].copy()

    #scale
    X_scaled = scaler.transform(df)

    #create time windows
    sequences = []
    for i in range(len(X_scaled) - timesteps):
        sequences.append(X_scaled[i:i + timesteps])
    
    X_seq = np.array(sequences)
    return X_seq

def feature_engineering_dispatcher_cnnlstm(df: pd.DataFrame, air_id: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"No data provided for AIR {air_id}")

    if air_id in ["12318", "Epi"]:
        df_mapped = map_influx_to_model_columns_12318(df)
        return feature_engineering_air12318(df_mapped)
    elif air_id in ["12300", "12305", "Military1", "Military2"]:
        df_mapped = map_influx_to_model_columns_123005(df)
        return feature_engineering_air12305(df_mapped)
    else:
        raise ValueError(f"CNN-LSTM feature engineering not defined for AIR {air_id}")



