# === tune_model.py ===
from Application.training.utils.imports import *

# Load and preprocess data
df = load_dataset()
df['TimeUntilNextWatering (hours)'] = df['TimeSinceLastWatering (hours)'].max() - df['TimeSinceLastWatering (hours)']
df['is_daytime'] = df['HourOfDay'].apply(lambda h: 1 if 6 <= h <= 18 else 0)
df['moisture_drop_rate'] = df['SoilMoisture (%)'].diff().fillna(0) * -1
df['part_of_day'] = df['HourOfDay'].apply(lambda h: 'morning' if h < 12 else 'afternoon' if h < 18 else 'evening')

# One-hot encode growth stage
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[['GrowthStage']])
growth_stage_cols = encoder.get_feature_names_out(['GrowthStage'])
df = pd.concat([df, pd.DataFrame(encoded, columns=growth_stage_cols)], axis=1)

# Define features and target
features = [
    'SoilMoisture (%)', 'AirTemperature (°C)', 'AirHumidity (%)',
    'LightLevel (lux)', 'HourOfDay', 'is_daytime', 'moisture_drop_rate'
] + list(growth_stage_cols)

X = df[features]
y = df['TimeUntilNextWatering (hours)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_dist = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

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

# Evaluation
mae = mean_absolute_error(y_test, best_model.predict(X_test))
r2 = r2_score(y_test, best_model.predict(X_test))
print(f"✅ Best Params: {search.best_params_}")
print(f"📊 MAE: {mae:.2f} | R²: {r2:.2f}")

# Save model + encoder
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_path = os.path.join("Application", "trained_models", f"tuned_model_{timestamp}.pkl")
encoder_path = os.path.join("Application", "trained_models", f"tuned_encoder_{timestamp}.pkl")


joblib.dump(best_model, model_path)
joblib.dump(encoder, encoder_path)
print(f"💾 Saved: {model_path}, {encoder_path}")

# Keep only latest 3 files
cleanup_old_files("Application/trained_models", "tuned_model_*.pkl", keep_last=3)
cleanup_old_files("Application/trained_models", "tuned_encoder_*.pkl", keep_last=3)

