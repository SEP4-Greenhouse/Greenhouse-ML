import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score

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

# === Feature selection ===
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

# === Hyperparameter Tuning ===
param_dist = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("🔍 Starting hyperparameter tuning...")
search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    scoring='neg_mean_absolute_error',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
print(f"✅ Best parameters found:\n{search.best_params_}")

# === Final evaluation ===
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Final MAE: {mae:.2f} hours")
print(f"📈 Final R² Score: {r2:.2f}")

# === Save tuned model and encoder ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_name = f"tuned_model_{timestamp}.pkl"
encoder_name = f"tuned_encoder_{timestamp}.pkl"

model_path = os.path.join(r"C:\Users\banra\college\semi 4\Sep4\Greenhouse-ML\Application\ml_model", model_name)
encoder_path = os.path.join(r"C:\Users\banra\college\semi 4\Sep4\Greenhouse-ML\Application\ml_model", encoder_name)

os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(best_model, model_path)
joblib.dump(encoder, encoder_path)

print(f"\n💾 Tuned model saved to: {model_path}")
print(f"💾 Encoder saved to: {encoder_path}")
