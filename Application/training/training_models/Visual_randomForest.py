from Application.training.utils.imports import *

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
    print(f"📦 Loaded encoder: {os.path.basename(encoder_files[0])}")
    print(f"📦 Loaded regressor: {os.path.basename(model_files[0])}")
    
    model = joblib.load(model_files[0])
    encoder = joblib.load(encoder_files[0])
    
    # === Load data for evaluation ===
    X, y, df, base_features = load_processed_dataset(encoder)
    print(df.columns.tolist())
    
    # === Add engineered features (fix SettingWithCopyWarning) ===
    X = X.copy()  # Create explicit copy to avoid warnings
    X.loc[:, "soil_dryness_index"] = 100 - X["Soil Humidity"]
    X.loc[:, "temp_soil_product"] = X["Temperature"] * X["Soil Humidity"]
    X.loc[:, "light_humidity_ratio"] = X["Light"] / (X["Air Humidity"] + 1)
    
    # === Train/test split (same as training) ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # === Make predictions ===
    y_pred = model.predict(X_test)
    
    # === Calculate metrics ===
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # === Visualizations ===
    # 1. Actual vs Predicted plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Hours')
    plt.ylabel('Predicted Hours')
    plt.title('Actual vs Predicted Hours Until Watering')  # Removed emoji to avoid font warning
    plt.tight_layout()
    
    # 2. Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.subplot(2, 2, 2)
    plt.bar(range(10), importances[indices][:10])
    plt.xticks(range(10), X.columns[indices][:10], rotation=90)
    plt.title('Feature Importance')  # Removed emoji to avoid font warning
    plt.tight_layout()
    
    # 3. Error Distribution
    errors = y_pred - y_test
    
    plt.subplot(2, 2, 3)
    plt.hist(errors, bins=20)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')  # Removed emoji to avoid font warning
    plt.tight_layout()
    
    # 4. Summary text
    plt.subplot(2, 2, 4)
    plt.text(0.5, 0.5, 
            f"Regression Evaluation Metrics\n"  # Removed emoji to avoid font warning
            f"   Mean Absolute Error (MAE): {mae:.2f} hours\n"
            f"   R² Score:                 {r2:.2f}",
            fontsize=12, horizontalalignment='center', verticalalignment='center')
    plt.axis('off')
    plt.tight_layout()
    
    plt.show()
    
except Exception as e:
    print(f"Error: {e}")