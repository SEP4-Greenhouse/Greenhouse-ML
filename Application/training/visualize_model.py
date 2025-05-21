import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# === Load dataset (same one used for training) ===
df = pd.read_csv("Greenhouse_data_MAL.csv")

# === Rebuild features like in training ===
df['TimeUntilNextWatering (hours)'] = df['TimeSinceLastWatering (hours)'].max() - df['TimeSinceLastWatering (hours)']
df['is_daytime'] = df['HourOfDay'].apply(lambda h: 1 if 6 <= h <= 18 else 0)
df['moisture_drop_rate'] = df['SoilMoisture (%)'].diff(periods=1).fillna(0) * -1

# One-hot encode GrowthStage using saved encoder
encoder_path = r"C:\Users\banra\college\semi 4\Sep4\Greenhouse-ML\Application\ml_model\tuned_encoder_2025-05-21_11-24-53.pkl"
encoder = joblib.load(encoder_path)
encoded_growth_stage = encoder.transform(df[['GrowthStage']])
growth_stage_cols = encoder.get_feature_names_out(['GrowthStage'])
df_encoded = pd.DataFrame(encoded_growth_stage, columns=growth_stage_cols)
df = pd.concat([df, df_encoded], axis=1)

# Features used
features = [
    'SoilMoisture (%)',
    'AirTemperature (°C)',
    'AirHumidity (%)',
    'LightLevel (lux)',
    'HourOfDay',
    'is_daytime',
    'moisture_drop_rate',
] + list(growth_stage_cols)

X = df[features]
y_true = df['TimeUntilNextWatering (hours)']

# === Load model ===
model_path = r"C:\Users\banra\college\semi 4\Sep4\Greenhouse-ML\Application\ml_model\tuned_model_2025-05-21_11-24-53.pkl"
model = joblib.load(model_path)

# === Predictions ===
y_pred = model.predict(X)
residuals = y_true - y_pred

# === Plot 1: Feature Importance ===
plt.figure(figsize=(10, 6))
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45, ha="right")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# === Plot 2: Actual vs Predicted ===
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.6)
plt.xlabel("Actual Time Until Watering (hours)")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 3: Residuals Histogram ===
plt.figure(figsize=(8, 4))
plt.hist(residuals, bins=20, edgecolor='black')
plt.title("Residual Error Distribution")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Print MAE and R² for full dataset ===
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(f"\n📊 Full Dataset MAE: {mae:.2f} hours")
print(f"📈 Full Dataset R²: {r2:.2f}")
