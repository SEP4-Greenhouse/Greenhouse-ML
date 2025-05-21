import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
import numpy as np
from datetime import datetime

# === Load dataset ===
df = pd.read_csv("Greenhouse_data_MAL.csv")

# === Feature Engineering ===
df['TimeUntilNextWatering (hours)'] = df['TimeSinceLastWatering (hours)'].max() - df['TimeSinceLastWatering (hours)']
df['is_daytime'] = df['HourOfDay'].apply(lambda h: 1 if 6 <= h <= 18 else 0)

def categorize_hour(hour):
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    else:
        return "evening"
df['part_of_day'] = df['HourOfDay'].apply(categorize_hour)

df['moisture_drop_rate'] = df['SoilMoisture (%)'].diff(periods=1).fillna(0) * -1

# === One-hot encode GrowthStage ===
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_growth_stage = encoder.fit_transform(df[['GrowthStage']])
growth_stage_cols = encoder.get_feature_names_out(['GrowthStage'])
df_encoded = pd.DataFrame(encoded_growth_stage, columns=growth_stage_cols)
df = pd.concat([df, df_encoded], axis=1)

# === Features and Target ===
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

# === Split dataset ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train model ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate model ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Model trained on split set")
print(f"📊 MAE: {mae:.2f} hours")
print(f"📈 R² Score: {r2:.2f}")

# === Cross-Validation ===
cv = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

print("\n🔁 Cross-Validation Results (5-fold):")
print(f"Avg MAE: {mae_scores.mean():.2f} | Std: {mae_scores.std():.2f}")
print(f"Avg R² : {r2_scores.mean():.2f} | Std: {r2_scores.std():.2f}")

# === Save model and encoder ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_name = f"model_{timestamp}.pkl"
encoder_name = f"growth_stage_encoder_{timestamp}.pkl"

model_path = os.path.join(r"C:\Users\banra\college\semi 4\Sep4\Greenhouse-ML\Application\ml_model", model_name)
encoder_path = os.path.join(r"C:\Users\banra\college\semi 4\Sep4\Greenhouse-ML\Application\ml_model", encoder_name)

os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
joblib.dump(encoder, encoder_path)

print(f"\n💾 Model saved to: {model_path}")
print(f"💾 Encoder saved to: {encoder_path}")
