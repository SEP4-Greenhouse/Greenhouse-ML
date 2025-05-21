import pandas as pd
import numpy as np
import joblib
import os
import glob
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# === Load dataset ===
df = pd.read_csv("Greenhouse_data_MAL.csv")

# === Feature Engineering ===
df['TimeUntilNextWatering (hours)'] = df['TimeSinceLastWatering (hours)'].max() - df['TimeSinceLastWatering (hours)']
df['is_daytime'] = df['HourOfDay'].apply(lambda h: 1 if 6 <= h <= 18 else 0)
df['moisture_drop_rate'] = df['SoilMoisture (%)'].diff(periods=1).fillna(0) * -1
df['part_of_day'] = df['HourOfDay'].apply(lambda h: 'morning' if h < 12 else 'afternoon' if h < 18 else 'evening')

# === One-hot encode GrowthStage ===
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_growth_stage = encoder.fit_transform(df[['GrowthStage']])
growth_stage_cols = encoder.get_feature_names_out(['GrowthStage'])
df_encoded = pd.DataFrame(encoded_growth_stage, columns=growth_stage_cols)
df = pd.concat([df, df_encoded], axis=1)

# === Features and target ===
features = [
    'SoilMoisture (%)',
    'AirTemperature (°C)',
    'AirHumidity (%)',
    'LightLevel (lux)',
    'HourOfDay',
    'is_daytime',
    'moisture_drop_rate',
] + list(growth_stage_cols)

target = 'TimeUntilNextWatering (hours)'
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train model ===
model = RandomForestRegressor(n_estimators=250, max_depth=10, min_samples_split=2, min_samples_leaf=4, random_state=42)
model.fit(X_train, y_train)

# === Evaluate model ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# === Create timestamped filenames ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_dir = r"C:\Users\banra\college\semi 4\Sep4\Greenhouse-ML\Application\ml_model"
model_path = os.path.join(model_dir, f"pipeline_model_{timestamp}.pkl")
encoder_path = os.path.join(model_dir, f"pipeline_encoder_{timestamp}.pkl")
log_path = os.path.join(model_dir, f"pipeline_log_{timestamp}.json")

# === Save model and encoder ===
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "model.pkl"))
joblib.dump(encoder, os.path.join(model_dir, "encoder.pkl"))


# === Save training log ===
log_data = {
    "timestamp": timestamp,
    "model_path": model_path,
    "encoder_path": encoder_path,
    "mae": round(mae, 3),
    "r2": round(r2, 3),
    "params": model.get_params()
}

# === Keep only latest 3 models/logs ===
def cleanup_old_files(directory, pattern, keep=3):
    files = sorted(glob.glob(os.path.join(directory, pattern)), key=os.path.getmtime, reverse=True)
    for f in files[keep:]:
        os.remove(f)

cleanup_old_files(model_dir, "pipeline_model_*.pkl", keep=3)
cleanup_old_files(model_dir, "pipeline_encoder_*.pkl", keep=3)
cleanup_old_files(model_dir, "pipeline_log_*.json", keep=3)

print("🧹 Cleaned up old model, encoder, and log files — only latest 3 kept.")

with open(log_path, 'w') as f:
    json.dump(log_data, f, indent=4)

print("✅ Model, encoder, and log saved:")
print(f" - Model: {model_path}")
print(f" - Encoder: {encoder_path}")
print(f" - Log: {log_path}")
print(f"📊 MAE: {mae:.2f} | R²: {r2:.2f}")
