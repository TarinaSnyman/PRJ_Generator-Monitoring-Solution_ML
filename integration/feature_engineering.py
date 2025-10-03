# feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

# helper function to map raw influx columns to model features 
def map_influx_to_model_columns(df: pd.DataFrame) -> pd.DataFrame:
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



# eventually other air feature engineering

#dispatches the feture engineered dataset for each air
def feature_engineering_dispatcher(df: pd.DataFrame, air_id: str) -> pd.DataFrame:
    # map Influx columns to model features
    df_mapped = map_influx_to_model_columns(df)
    if air_id == "12318" or air_id=="Epi":
        return feature_engineering_air12318(df_mapped)
    # elif air_id == "":
    #     return feature_engineering_air56789(df_mapped)
    else:
        raise ValueError(f"Feature engineering not defined for AIR {air_id}")
