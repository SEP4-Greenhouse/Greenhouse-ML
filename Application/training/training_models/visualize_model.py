# === visualize_model.py ===
from Application.training.utils.imports import *
import os

# === Constants ===
TRAINED_MODEL_DIR = os.path.join("Application", "trained_models")
MODEL_PATTERN = os.path.join(TRAINED_MODEL_DIR, "tuned_model_*.pkl")
ENCODER_PATTERN = os.path.join(TRAINED_MODEL_DIR, "tuned_encoder_*.pkl")

# === Load latest model & encoder ===
def get_latest(path_pattern):
    files = sorted(glob.glob(path_pattern), key=os.path.getmtime, reverse=True)
    return files[0] if files else None

model_path = get_latest(MODEL_PATTERN)
encoder_path = get_latest(ENCODER_PATTERN)

if model_path is None or encoder_path is None:
    raise FileNotFoundError("Model or encoder file not found in Application/trained_models/")

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)

# === Load and preprocess dataset ===
X, y_true, df, _ = load_processed_dataset(encoder=encoder)

# Ensure correct column order and matching names
expected_features = list(model.feature_names_in_)
X = pd.DataFrame(X, columns=X.columns)
X = X.reindex(columns=expected_features)

# === Predict & Evaluate ===
y_pred = model.predict(X)
residuals = y_true - y_pred

# === Plot 1: Feature Importance ===
plt.figure(figsize=(10, 6))
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [model.feature_names_in_[i] for i in indices], rotation=45, ha="right")
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

# === Print metrics ===
print(f"\n📊 Full MAE: {mean_absolute_error(y_true, y_pred):.2f}")
print(f"📈 Full R²: {r2_score(y_true, y_pred):.2f}")
