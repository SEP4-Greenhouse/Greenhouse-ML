# === helpers.py ===
from Application.training.utils.imports import *

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def cleanup_old_files(directory, pattern, keep=3):
    """Delete all but latest N files matching pattern."""
    files = sorted(glob.glob(os.path.join(directory, pattern)), key=os.path.getmtime, reverse=True)
    for f in files[keep:]:
        os.remove(f)