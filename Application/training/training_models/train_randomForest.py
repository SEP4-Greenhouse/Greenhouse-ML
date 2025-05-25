# === Imports ===
from Application.training.utils.imports import *
from Application.training.utils.file_manager import get_timestamp, save_model, save_log, cleanup_old_files
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os
import joblib

# === Constants ===
MODEL_DIR = os.path.join("Application", "trained_models")
LOG_DIR = os.path.join(MODEL_DIR, "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Load processed dataset ===
X, y, df, _ = load_processed_dataset()  # Don't use encoder here

# === Use only the 16 features expected by the prediction service ===
X = X.copy()
X["co2"] = 400.0         # Fill with constant default (if not in dataset)
X["pir"] = 0.0
X["proximity"] = 0.0

# Manually map plantGrowthStage to numbers
stage_map = {
    "Seedling": 0, "Seedling Stage": 0,
    "Vegetative": 1, "Vegetative Stage": 1,
    "Flowering": 2, "Flowering Stage": 2
}
X["growth_stage"] = df["plantGrowthStage"].map(stage_map).fillna(1)

# Engineered features
X["temp_soil"] = X["Temperature"] * X["Soil Humidity"] / 100.0
X["temp_air"] = X["Temperature"] * X["Air Humidity"] / 100.0
X["light_temp"] = X["Light"] * X["Temperature"] / 1000.0
X["soil_air"] = X["Soil Humidity"] * X["Air Humidity"] / 100.0
X["time_soil"] = X["timeSinceLastWateringInHours"] * X["Soil Humidity"] / 100.0
X["temp_squared"] = X["Temperature"] ** 2 / 100.0
X["soil_squared"] = X["Soil Humidity"] ** 2 / 100.0

# === Final 16 feature columns ===
features = [
    "Temperature", "Soil Humidity", "Air Humidity", "Light", "co2", "pir", "proximity",
    "timeSinceLastWateringInHours", "growth_stage",
    "temp_soil", "temp_air", "light_temp", "soil_air", "time_soil",
    "temp_squared", "soil_squared"
]

X = X[features]

# === Split and train ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"📊 MAE = {mae:.3f} | R² = {r2:.3f}")

# === Save model ===
timestamp = get_timestamp()
model_path, _ = save_model(model, None, timestamp, prefix="reg_")

# === Save log ===
log_data = {
    "timestamp": timestamp,
    "regressor_path": model_path,
    "regression_mae": round(mae, 3),
    "regression_r2": round(r2, 3),
    "features_used": features
}
save_log(log_data, timestamp, prefix="regression_only_")

# === Cleanup old models/logs ===
cleanup_old_files(MODEL_DIR, "reg_model_*.pkl", keep_last=1)
cleanup_old_files(LOG_DIR, "regression_only_log_*.json", keep_last=1)
