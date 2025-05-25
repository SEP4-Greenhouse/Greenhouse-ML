# === Imports ===
from Application.training.utils.imports import *

# === Setup encoder ===
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# === Load processed dataset ===
X, y, df, base_features = load_processed_dataset(encoder)

# === Feature engineering ===
X = X.copy()
X["soil_dryness_index"] = 100 - X["Soil Humidity"]
X["temp_soil_product"] = X["Temperature"] * X["Soil Humidity"]
X["light_humidity_ratio"] = X["Light"] / (X["Air Humidity"] + 1)

# === Final feature list ===
stage_cols = encoder.get_feature_names_out(["plantGrowthStage"])
features = base_features + ["soil_dryness_index", "temp_soil_product", "light_humidity_ratio"] + list(stage_cols)
X = X[features]

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train model with fixed parameters ===
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

# === Save model and encoder ===
timestamp = get_timestamp()
reg_path, encoder_path = save_model(model, encoder, timestamp, prefix="reg_")

# === Log ===
log_data = {
    "timestamp": timestamp,
    "regressor_path": reg_path,
    "encoder_path": encoder_path,
    "regression_mae": round(mae, 3),
    "regression_r2": round(r2, 3),
    "features_used": features
}
save_log(log_data, timestamp, prefix="regression_only_")

# === Cleanup ===
cleanup_old_files(MODEL_DIR, "reg_model_*.pkl", keep_last=1)
cleanup_old_files(MODEL_DIR, "reg_encoder_*.pkl", keep_last=1)
cleanup_old_files(LOG_DIR, "regression_only_log_*.json", keep_last=1)
