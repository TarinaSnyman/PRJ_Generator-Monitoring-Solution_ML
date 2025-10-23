import os
import pandas as pd
import numpy as np

# Path configuration
input_folder = "./"
output_folder = "../onFailingData"
os.makedirs(output_folder, exist_ok=True)

def simulate_failure(df, air_id):
    df = df.copy()
    df = df.apply(pd.to_numeric, errors="ignore")

    # Create a time factor from 0 → 1 over the dataset length
    t = np.linspace(0, 1, len(df))

    # 12318 — voltage imbalance and PF degradation
    if air_id in ["12318", "Epi"]:
        for col, end_scale in zip(["ia_A", "ib_A", "ic_A"], [0.9, 0.8, 0.7]):
            if col in df.columns:
                df[col] *= np.linspace(1, end_scale, len(df))
        if "pftot_None" in df.columns:
            df["pftot_None"] -= 0.2 * t  # degrade PF
        if "ptot_W" in df.columns:
            df["ptot_W"] *= np.linspace(1, 0.85, len(df))  # drop total power

    # 12300 — oil pressure & temperature rise, RPM unstable
    elif air_id in ["12300", "Military1"]:
        if "oilpress_pascal" in df.columns:
            df["oilpress_pascal"] += 50000 * t
        if "cooltemp_degree-celsius" in df.columns:
            df["cooltemp_degree-celsius"] += 15 * t
        if "rpm_revolutions-per-minute" in df.columns:
            df["rpm_revolutions-per-minute"] *= 1 + 0.05 * np.sin(10 * t * np.pi)

    # 12305 — frequency and voltage sag
    elif air_id in ["12305", "Military2"]:
        for col in ["va_volt", "vb_volt", "vc_volt", "vlineavg_volt"]:
            if col in df.columns:
                df[col] *= np.linspace(1, 0.85, len(df))
        if "freq_hertz" in df.columns:
            df["freq_hertz"] *= np.linspace(1, 0.95, len(df))

    else:
        print(f"⚠️ Unknown air_id: {air_id}. No specific failure pattern applied.")

    # Add some noise for realism
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] += np.random.normal(0, 0.005 * df[col].mean(), len(df))

    return df


def main():
    for file in os.listdir(input_folder):
        if not file.endswith(".csv"):
            continue

        try:
            air_id = file.split("_")[0].replace("air", "")
            path = os.path.join(input_folder, file)
            df = pd.read_csv(path)
            print(f"✅ Processing {file} for AIR {air_id}")

            df_failed = simulate_failure(df, air_id)
            output_path = os.path.join(output_folder, file.replace(".csv", "_FailingData.csv"))
            df_failed.to_csv(output_path, index=False)
            print(f"   ↳ Saved simulated failing data to: {output_path}")

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")


if __name__ == "__main__":
    main()
