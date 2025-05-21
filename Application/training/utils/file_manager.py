# === file_manager.py ===
from Application.training.utils.imports import *

MODEL_DIR = os.path.join("Application","trained_models")
LOG_DIR = os.path.join(MODEL_DIR, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def save_model(model, encoder, timestamp):
    model_path = os.path.join(MODEL_DIR, f"pipeline_model_{timestamp}.pkl")
    encoder_path = os.path.join(MODEL_DIR, f"pipeline_encoder_{timestamp}.pkl")
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    return model_path, encoder_path

def save_log(log_data, timestamp):
    log_path = os.path.join(LOG_DIR, f"pipeline_log_{timestamp}.json")
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)
    return log_path

def cleanup_old_files(folder: str, pattern: str, keep_last: int = 3):
    files = sorted(glob.glob(os.path.join(folder, pattern)), key=os.path.getmtime, reverse=True)
    for file in files[keep_last:]:
        os.remove(file)