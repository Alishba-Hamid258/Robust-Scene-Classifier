import tensorflow as tf
import os
import numpy as np
from config import MODEL_DIR, DATA_DIR, LABELS

def evaluate_model(model_path, test_dir, needs_rescaling=False):
    print(f"\n--- Evaluating {os.path.basename(model_path)} (Rescale: {needs_rescaling}) ---")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return 0
    
    # Load test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(224, 224),
        batch_size=32,
        label_mode='categorical',
        shuffle=True
    )

    if needs_rescaling:
        test_ds = test_ds.map(lambda x, y: (x / 255.0, y))
    
    results = model.evaluate(test_ds.take(20))
    print(f"Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")
    return results[1]

test_path = os.path.join(DATA_DIR, "seg_test", "seg_test")
models_config = [
    {"path": os.path.join(MODEL_DIR, "best_model_final.keras"), "rescale": False},
    {"path": os.path.join(MODEL_DIR, "best_model_vit.keras"), "rescale": True}, # ViT usually needs 0-1
    {"path": os.path.join(MODEL_DIR, "best_model.h5"), "rescale": True}
]

for m in models_config:
    if os.path.exists(m["path"]):
        evaluate_model(m["path"], test_path, m["rescale"])
    else:
        print(f"Model not found: {m['path']}")
