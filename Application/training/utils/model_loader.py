import os
import pickle
import joblib
import traceback
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def safe_load_model(model_path, default_n_estimators=50, default_max_depth=5):
    """
    Load a model with fallback mechanisms for NumPy version incompatibilities.
    
    This function tries multiple approaches to load a model that might have been
    saved with a different NumPy version.
    
    Args:
        model_path: Path to the saved model file (.pkl)
        default_n_estimators: Number of estimators for fallback model (default: 50)
        default_max_depth: Maximum depth for fallback model (default: 5)
        
    Returns:
        Loaded model or fallback model if loading fails
        
    Raises:
        Exception: If the error is not related to NumPy compatibility
    """
    if not os.path.exists(model_path):
        print(f"[MODEL_LOADER] Error: Model path does not exist: {model_path}")
        print(f"[MODEL_LOADER] Creating fallback model")
        return RandomForestRegressor(n_estimators=default_n_estimators, max_depth=default_max_depth)
        
    try:
        # Try direct loading first - the normal approach
        model = joblib.load(model_path)
        print(f"[MODEL_LOADER] Successfully loaded model from {model_path}")
        return model
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e):
            print(f"[MODEL_LOADER] NumPy version incompatibility detected, trying alternative loading method")
            try:
                # Try pickle with latin1 encoding - can help with NumPy version issues
                with open(model_path, 'rb') as f:
                    model = pickle.load(f, encoding='latin1')
                print(f"[MODEL_LOADER] Successfully loaded model with pickle fallback")
                return model
            except Exception as pickle_error:
                # If pickle also fails, log the error
                print(f"[MODEL_LOADER] Pickle loading failed: {str(pickle_error)}")
                print(f"[MODEL_LOADER] Error details: {traceback.format_exc()}")
                
                # Create a very basic random forest as last resort
                print(f"[MODEL_LOADER] Creating simplified replacement model")
                return RandomForestRegressor(n_estimators=default_n_estimators, max_depth=default_max_depth) 
        else:
            # If it's not a NumPy issue, re-raise the exception
            raise
    except Exception as e:
        # Catch any other exceptions during model loading
        print(f"[MODEL_LOADER] Unexpected error loading model: {str(e)}")
        print(f"[MODEL_LOADER] Error details: {traceback.format_exc()}")
        print(f"[MODEL_LOADER] Creating fallback model")
        return RandomForestRegressor(n_estimators=default_n_estimators, max_depth=default_max_depth)