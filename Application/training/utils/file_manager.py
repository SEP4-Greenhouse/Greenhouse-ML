import os
import json
import joblib
import glob
from datetime import datetime

# Directory setup for models and logs
MODEL_DIR = os.path.join("Application", "trained_models")
LOG_DIR = os.path.join(MODEL_DIR, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def get_timestamp():
    """Get current timestamp formatted as string for filenames."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_model(model, encoder, timestamp=None, prefix="pipeline_"):
    """
    Save machine learning model and encoder with timestamp.

    Args:
        model: Trained ML model
        encoder: One-hot encoder for categorical variables
        timestamp: Timestamp string for file naming (if None, generates new timestamp)
        prefix: Filename prefix (default: 'pipeline_')

    Returns:
        tuple: (model_path, encoder_path)
    """
    if timestamp is None:
        timestamp = get_timestamp()

    model_path = os.path.join(MODEL_DIR, f"{prefix}model_{timestamp}.pkl")
    encoder_path = os.path.join(MODEL_DIR, f"{prefix}encoder_{timestamp}.pkl")
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    print(f"Model saved to: {model_path}")
    print(f"Encoder saved to: {encoder_path}")
    return model_path, encoder_path

def save_log(log_data, timestamp=None, prefix="pipeline_"):
    """
    Save training logs as JSON with timestamp.

    Args:
        log_data: Dictionary containing log information
        timestamp: Timestamp string for file naming (if None, generates new timestamp)
        prefix: Filename prefix (default: 'pipeline_')

    Returns:
        str: Path to saved log file
    """
    if timestamp is None:
        timestamp = get_timestamp()

    log_path = os.path.join(LOG_DIR, f"{prefix}log_{timestamp}.json")
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f"Log saved to: {log_path}")
    return log_path

def cleanup_old_files(folder, pattern, keep_last=3):
    """
    Delete older files, keeping only the most recent ones.

    Args:
        folder: Directory containing files
        pattern: Glob pattern to match files
        keep_last: Number of recent files to keep (default: 3)
    """
    files = sorted(glob.glob(os.path.join(folder, pattern)), key=os.path.getmtime, reverse=True)
    for file in files[keep_last:]:
        os.remove(file)
        print(f"Deleted: {file}")

def load_latest_model(model_prefix="reg_model_"):
    """
    Load the latest trained model.

    Args:
        model_prefix: Prefix for model files (default: 'reg_model_')

    Returns:
        tuple: (model, encoder, model_path, encoder_path)
    """
    pattern = os.path.join(MODEL_DIR, f"{model_prefix}*.pkl")
    model_files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

    if not model_files:
        raise FileNotFoundError(f"No model files found matching {pattern}")

    model_path = model_files[0]
    encoder_prefix = model_prefix.replace('model', 'encoder')
    encoder_pattern = os.path.join(MODEL_DIR, f"{encoder_prefix}*.pkl")
    encoder_files = sorted(glob.glob(encoder_pattern), key=os.path.getmtime, reverse=True)

    if not encoder_files:
        raise FileNotFoundError(f"No encoder files found matching {encoder_pattern}")

    encoder_path = encoder_files[0]

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    print(f"Loaded model: {model_path}")
    print(f"Loaded encoder: {encoder_path}")

    return model, encoder, model_path, encoder_path

def load_latest_features(features_prefix="reg_features_"):
    """
    Load the latest feature list used for model training.

    Args:
        features_prefix: Prefix for saved feature files (default: 'reg_features_')

    Returns:
        list: Features used for training
    """
    pattern = os.path.join(MODEL_DIR, f"{features_prefix}*.pkl")
    feature_files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not feature_files:
        raise FileNotFoundError(f"No features file found matching {pattern}")
    latest_feature_file = feature_files[0]
    return joblib.load(latest_feature_file)
