import os

# ==========================================
# PATH CONFIGURATION
# ==========================================
# Get the absolute path of the project directory (where config.py lives)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Use Environment Variables for portability. 
# Defaults to C:\ML_Work for backward compatibility with existing data migration.
BASE_LOCAL_PATH = os.getenv("ML_WORK_DIR", r"C:\ML_Work")

# Cross-platform fallback: If C:\ doesn't exist (e.g., on Mac/Linux/Cloud),
# or we don't have permission, fallback to a folder inside the project root.
if not os.path.exists(BASE_LOCAL_PATH):
    try:
        os.makedirs(BASE_LOCAL_PATH, exist_ok=True)
    except (OSError, PermissionError):
        BASE_LOCAL_PATH = os.path.join(PROJECT_ROOT, "local_ml_data")
        os.makedirs(BASE_LOCAL_PATH, exist_ok=True)

# Data Paths
DATA_DIR = os.path.join(BASE_LOCAL_PATH, "datasets")
TRAIN_DIR = os.path.join(DATA_DIR, "seg_train", "seg_train")
TEST_DIR = os.path.join(DATA_DIR, "seg_test", "seg_test")
PRED_DIR = os.path.join(DATA_DIR, "seg_pred", "seg_pred")

# Model Paths
MODEL_DIR = os.path.join(BASE_LOCAL_PATH, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------
# ACTIVE MODEL CONFIGURATION
# ---------------------------------------------------------
# Model: best_model.h5
# Accuracy: ~92.3%
# Preprocessing Required: 
#   - Resize to (224, 224)
#   - Rescale pixel values to [0, 1] (divide by 255.0)
#   - RGB format
# ---------------------------------------------------------
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")

# Labels
LABELS = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'other', 5: 'sea', 6: 'street'}

def get_config_summary():
    return f"""
[ML PROJECT CONFIGURATION]
---------------------------
Local Base: {BASE_LOCAL_PATH}
Data Dir:   {DATA_DIR}
Model Path: {BEST_MODEL_PATH}
---------------------------
Status: READY
"""

if __name__ == "__main__":
    print(get_config_summary())
