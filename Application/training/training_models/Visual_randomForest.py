from Application.training.utils.imports import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# === Load latest model and encoder ===
try:
    # Get latest model files
    model_files = sorted(glob.glob(os.path.join(MODEL_DIR, "reg_model_*.pkl")), 
                        key=os.path.getmtime, reverse=True)
    encoder_files = sorted(glob.glob(os.path.join(MODEL_DIR, "reg_encoder_*.pkl")), 
                          key=os.path.getmtime, reverse=True)
    
    if not model_files or not encoder_files:
        raise FileNotFoundError("Model or encoder files not found")
        
    # Load the model and encoder
    print(f"Loaded encoder: {os.path.basename(encoder_files[0])}")
    print(f"Loaded regressor: {os.path.basename(model_files[0])}")
    
    model = joblib.load(model_files[0])
    encoder = joblib.load(encoder_files[0])
    
    # === Load data for evaluation ===
    X, y, df, base_features = load_processed_dataset(encoder)
    print(f"Dataset loaded with {len(X)} samples and {len(base_features)} base features")
    
    # === Add engineered features (fix SettingWithCopyWarning) ===
    X = X.copy()  # Create explicit copy to avoid warnings
    X.loc[:, "soil_dryness_index"] = 100 - X["Soil Humidity"]
    X.loc[:, "temp_soil_product"] = X["Temperature"] * X["Soil Humidity"]
    X.loc[:, "light_humidity_ratio"] = X["Light"] / (X["Air Humidity"] + 1)
    
    # === Train/test split (same as training) ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Test set: {len(X_test)} samples")
    
    # === Make predictions ===
    y_pred = model.predict(X_test)
    
    # === Calculate metrics ===
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {mae:.3f} hours | RMSE: {rmse:.3f} hours | R²: {r2:.3f}")
    
    # === Create figure with custom style ===
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 10))
    
    # === Visualizations ===
    # 1. Actual vs Predicted plot
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', linewidths=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Hours Until Watering', fontsize=10)
    plt.ylabel('Predicted Hours Until Watering', fontsize=10)
    plt.title('Actual vs Predicted Hours', fontsize=12, fontweight='bold')
    
    # 2. Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.subplot(2, 2, 2)
    top_n = min(10, len(X.columns))  # Ensure we don't exceed column count
    feature_names = [X.columns[i] for i in indices[:top_n]]
    plt.barh(range(top_n), importances[indices[:top_n]], color='skyblue')
    plt.yticks(range(top_n), feature_names)
    plt.xlabel('Importance Score')
    plt.title('Top Features by Importance', fontsize=12, fontweight='bold')
    
    # 3. Error Distribution
    errors = y_pred - y_test
    
    plt.subplot(2, 2, 3)
    plt.hist(errors, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error (hours)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution', fontsize=12, fontweight='bold')
    
    # 4. Summary text
    plt.subplot(2, 2, 4)
    metrics_text = (
        f"Model Evaluation Metrics\n\n"
        f"Mean Absolute Error (MAE):\n   {mae:.2f} hours\n\n"
        f"Root Mean Squared Error (RMSE):\n   {rmse:.2f} hours\n\n"
        f"R² Score:\n   {r2:.3f}\n\n"
        f"Test Samples: {len(X_test)}"
    )
    plt.text(0.5, 0.5, metrics_text, fontsize=11, 
             horizontalalignment='center', verticalalignment='center', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor='aliceblue', alpha=0.5))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
except Exception as e:
    print(f"Error visualizing model: {e}")
    import traceback
    traceback.print_exc()