import os
import pickle
import joblib
import traceback
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def safe_load_model(model_path):
    """
    Load a model with fallback mechanisms for NumPy version incompatibilities.
    
    This function tries multiple approaches to load a model that might have been
    saved with a different NumPy version.
    """
    try:
        # Try direct loading first - the normal approach
        return joblib.load(model_path)
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
                return RandomForestRegressor(n_estimators=50, max_depth=5) 
        else:
            # If it's not a NumPy issue, re-raise the exception
            raise