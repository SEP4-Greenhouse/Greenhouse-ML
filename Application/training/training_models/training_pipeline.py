# === training_pipeline.py ===
from Application.training.utils.imports import *


# === Load and preprocess dataset ===
df = load_dataset()
df['TimeUntilNextWatering (hours)'] = df['TimeSinceLastWatering (hours)'].max() - df['TimeSinceLastWatering (hours)']
df['is_daytime'] = df['HourOfDay'].apply(lambda h: 1 if 6 <= h <= 18 else 0)
df['moisture_drop_rate'] = df['SoilMoisture (%)'].diff().fillna(0) * -1
df['part_of_day'] = df['HourOfDay'].apply(lambda h: 'morning' if h < 12 else 'afternoon' if h < 18 else 'evening')

# One-hot encode GrowthStage
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[['GrowthStage']])
growth_stage_cols = encoder.get_feature_names_out(['GrowthStage'])
df = pd.concat([df, pd.DataFrame(encoded, columns=growth_stage_cols)], axis=1)

features = ['SoilMoisture (%)', 'AirTemperature (°C)', 'AirHumidity (%)', 'LightLevel (lux)', 'HourOfDay', 'is_daytime', 'moisture_drop_rate'] + list(growth_stage_cols)
X = df[features]
y = df['TimeUntilNextWatering (hours)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=250, max_depth=10, min_samples_split=2, min_samples_leaf=4, random_state=42)
model.fit(X_train, y_train)

mae = mean_absolute_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))

timestamp = get_timestamp()

model_path, encoder_path = save_model(model, encoder, timestamp)
# Also save latest version to Application/ml_model/ for the ML service
default_model_dir = "Application/trained_models"
os.makedirs(default_model_dir, exist_ok=True)

joblib.dump(model, os.path.join(default_model_dir, "model.pkl"))
joblib.dump(encoder, os.path.join(default_model_dir, "encoder.pkl"))

log_path = save_log({
    "timestamp": timestamp,
    "model_path": model_path,
    "encoder_path": encoder_path,
    "mae": round(mae, 3),
    "r2": round(r2, 3),
    "params": model.get_params()
}, timestamp)

cleanup_old_files("Application/trained_models", "pipeline_model_*.pkl")
cleanup_old_files("Application/trained_models", "pipeline_encoder_*.pkl")
cleanup_old_files("Application/trained_models/logs", "pipeline_log_*.json")

print("✅ Model, encoder, and log saved.")
