
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageStat
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import logging
import io
from config import BEST_MODEL_PATH, LABELS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="SceneAI Universal Pro")

# PATH TO OUR NEW FINAL MODEL
model_path = BEST_MODEL_PATH

try:
    logger.info("Loading Optimized EfficientNetV2 Model...")
    model = tf.keras.models.load_model(model_path)
    logger.info("Model is ONLINE and STABLE!")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise Exception(f"Model loading failed. Please ensure '{os.path.basename(model_path)}' exists at {model_path}. Error: {str(e)}")

# LABELS are imported from config

def calculate_complexity(image: Image.Image):
    grayscale = image.convert('L')
    stat = ImageStat.Stat(grayscale)
    stddev = stat.stddev[0]
    
    arr = np.array(grayscale)
    
    # SHANNON ENTROPY: The ultimate mathematical measure of image "chaos"
    # Natural landscapes have high entropy (7.0 to 8.0)
    # Sketches, documents, and flat shapes have low entropy (1.0 to 5.5)
    hist, _ = np.histogram(arr, bins=256, range=(0, 255))
    hist = hist / np.sum(hist)
    entropy = -np.sum([p * np.log2(p) for p in hist if p > 0])
    
    # 1. Sketch & Document Detection via Entropy
    if entropy < 5.5: 
        return True, f"Sketch/Document detected (Low Entropy: {entropy:.2f})"
        
    # 2. Low Contrast: Completely washed out/dark image
    if stddev < 12: 
        return True, "Extremely Low Contrast"
        
    return False, None

def preprocess_image(image: Image.Image):
    image = ImageOps.exif_transpose(image)
    img = image.resize((224, 224), Image.LANCZOS)  
    img_array = np.array(img).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        is_suspicious, suspicion_reason = calculate_complexity(pil_image)
        
        # 🛡️ THE SKETCH-BLOCKER: If it's a sketch/document, force it to 'other' instantly
        if is_suspicious and ("Sketch" in suspicion_reason or "Document" in suspicion_reason):
            scores = {val: 0.0 for val in LABELS.values()}
            scores['other'] = 1.0
            return {
                "predicted_class": "other",
                "confidence": 1.0,
                "confidence_gap": 1.0,
                "class_probabilities": scores,
                "suspicion": suspicion_reason,
                "is_confused": False
            }
        
        img_array = preprocess_image(pil_image)
        predictions = model.predict(img_array)[0]
        
        sorted_indices = np.argsort(predictions)[::-1]
        top_1_idx = sorted_indices[0]
        top_2_idx = sorted_indices[1]
        confidence_gap = float(predictions[top_1_idx] - predictions[top_2_idx])
        
        pred_class = LABELS[top_1_idx]
        confidence = float(predictions[top_1_idx])
        scores = {LABELS[i]: float(predictions[i]) for i in range(len(LABELS))}
        
        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "confidence_gap": confidence_gap,
            "class_probabilities": scores,
            "suspicion": suspicion_reason if is_suspicious else None,
            "is_confused": bool(confidence_gap < 0.10)
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
