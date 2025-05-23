from Application.training.utils.imports import *

# === Load and process data using data_loader ===
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X, y, df, base_features = load_processed_dataset(encoder)

# === Add engineered features (avoid SettingWithCopyWarning) ===
X = X.copy()  # Create an explicit copy to avoid warnings
X.loc[:, "soil_dryness_index"] = 100 - X["Soil Humidity"]
X.loc[:, "temp_soil_product"] = X["Temperature"] * X["Soil Humidity"]
X.loc[:, "light_humidity_ratio"] = X["Light"] / (X["Air Humidity"] + 1)

# === Define complete features list ===
stage_cols = encoder.get_feature_names_out(["plantGrowthStage"])
features = base_features + ["soil_dryness_index", "temp_soil_product", "light_humidity_ratio"] + list(stage_cols)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Define parameter grid ===
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': [0.5, 0.7, 0.8, 'sqrt', 'log2'],
}

# === Run RandomizedSearchCV ===
base_model = RandomForestRegressor(random_state=42)
rs_cv = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("Fitting 5 folds for each of 20 candidates, totalling 100 fits")
rs_cv.fit(X_train, y_train)

# === Get best model ===
best_params = rs_cv.best_params_
print(f"✅ Best Params: {best_params}")

# === Evaluate ===
best_model = rs_cv.best_estimator_
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"📊 Regression MAE: {mae:.3f} | R²: {r2:.3f}")

# === Save model and encoder ===
timestamp = get_timestamp()
reg_path, encoder_path = save_model(best_model, encoder, timestamp, prefix="reg_")

# === Save logs ===
log_data = {
    "timestamp": timestamp,
    "regressor_path": reg_path,
    "encoder_path": encoder_path,
    "best_params": {str(k): (str(v) if isinstance(v, (list, dict)) else v) for k, v in best_params.items()},
    "regression_mae": round(mae, 3),
    "regression_r2": round(r2, 3),
    "features_used": features,
}
log_path = save_log(log_data, timestamp, prefix="regression_tuned_")

# === Cleanup old files ===
cleanup_old_files(MODEL_DIR, "reg_model_*.pkl", keep_last=1)
cleanup_old_files(MODEL_DIR, "reg_encoder_*.pkl", keep_last=1)
cleanup_old_files(LOG_DIR, "regression_only_log_*.json", keep_last=1)
cleanup_old_files(LOG_DIR, "regression_tuned_log_*.json", keep_last=1)