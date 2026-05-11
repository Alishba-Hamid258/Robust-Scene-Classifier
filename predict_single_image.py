
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
model_path = r"C:\Users\USER\Documents\main\best_model.h5"
image_path = r"C:\Users\USER\Documents\main_cleaned\seg_pred\seg_pred\5.jpg"  # Replace with your image path

def load_and_preprocess_image(image_path):
    logger.info(f"Loading and preprocessing image: {image_path}")
    if not os.path.exists(image_path):
        logger.error(f"Image not found at {image_path}")
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Load image
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to match training input
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    
    # Ensure 3 channels (RGB)
    if img_array.ndim == 2:  # Grayscale to RGB
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # Remove alpha channel if present
        img_array = img_array[:, :, :3]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

def predict_image(model, img_array):
    logger.info("Making prediction...")
    predictions = model.predict(img_array)
    pred_class_idx = np.argmax(predictions[0])
    labels = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}
    pred_class = labels[pred_class_idx]
    confidence = predictions[0][pred_class_idx]
    
    # Log confidence scores for all classes
    logger.info("Confidence scores:")
    for idx, label in labels.items():
        logger.info(f"{label}: {predictions[0][idx]:.4f}")
    
    return pred_class, confidence, predictions[0]

def display_image(img, pred_class, confidence):
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {pred_class} ({confidence:.4f})")
    plt.axis('off')
    plt.show()

def main():
    # Verify model
    logger.info("Verifying model file...")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load model
    logger.info("Loading trained model...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise
    
    # Load and preprocess image
    img, img_array = load_and_preprocess_image(image_path)
    
    # Predict
    pred_class, confidence, scores = predict_image(model, img_array)
    
    # Display result
    logger.info(f"Predicted class: {pred_class} (Confidence: {confidence:.4f})")
    display_image(img, pred_class, confidence)

if __name__ == "__main__":
    main()
