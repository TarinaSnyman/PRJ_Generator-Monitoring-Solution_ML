# feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

def feature_engineering_air12318(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs ON/OFF detection and feature engineering for AIR 12318.
    Returns a dataframe of engineered features ready for modeling.
    """
    # --- 1. ON/OFF detection ---
    p = df['ptot_W'].clip(lower=0).fillna(0)
    X = np.log1p(p).values.reshape(-1,1)
    gmm = GaussianMixture(n_components=2, random_state=42).fit(X)
    labels = gmm.predict(X)
    means = gmm.means_.flatten()
    on_cluster = np.argmax(means)
    df['is_running_gmm'] = (labels == on_cluster).astype(int)

    df['any_current'] = df[['ia_A','ib_A','ic_A']].sum(axis=1) > 0.1
    df['is_running'] = ((df['is_running_gmm']==1) | df['any_current']).astype(int)

    # Filter ON cycles
    df_on = df[df['is_running']==1].copy()

    # --- 2. Feature engineering ---
    if all(col in df_on.columns for col in ['ia_A','ib_A','ic_A']):
        df_on['current_imbalance'] = df_on[['ia_A','ib_A','ic_A']].std(axis=1)

    if 'pftot_None' in df_on.columns:
        df_on['pf_anomaly'] = np.abs(1 - df_on['pftot_None'])

    temp_cols = [c for c in df_on.columns if 'temp' in c.lower()]
    for col in temp_cols:
        df_on[f'{col}_roc'] = df_on[col].diff()

    fuel_cols = [c for c in df_on.columns if 'fuel' in c.lower()]
    for col in fuel_cols:
        df_on[f'{col}_roc'] = df_on[col].diff()

    rolling_window = 60
    for col in ['ptot_W','ia_A','pf_anomaly']:
        if col in df_on.columns:
            df_on[f'{col}_rollmean'] = df_on[col].rolling(window=rolling_window, min_periods=1).mean()
            df_on[f'{col}_rollstd']  = df_on[col].rolling(window=rolling_window, min_periods=1).std()

    # --- 3. Select feature columns ---
    exclude_cols = ['time','is_running','iforest_anomaly','heuristic_anomaly']
    feature_cols = [c for c in df_on.columns if c not in exclude_cols]

    return df_on[feature_cols]


# eventually other air feature engineering

#dispatches the feture engineered dataset for each air
def feature_engineering_dispatcher(df: pd.DataFrame, air_id: str) -> pd.DataFrame:
    if air_id == "12318":
        return feature_engineering_air12318(df)
    # elif air_id == "56789":
    #     return feature_engineering_air56789(df)
    else:
        raise ValueError(f"Feature engineering not defined for AIR {air_id}")
