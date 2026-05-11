
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model_path = r"C:\Users\USER\Documents\main\best_model.h5"
pred_path = r"C:\Users\USER\Documents\main_cleaned\seg_pred\seg_pred"

def load_and_preprocess_image(image_path):
    logger.info(f"Loading and preprocessing image: {image_path}")
    if not os.path.exists(image_path):
        logger.error(f"Image not found at {image_path}")
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

def predict_image(model, img_array):
    predictions = model.predict(img_array)
    pred_class_idx = np.argmax(predictions[0])
    labels = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}
    pred_class = labels[pred_class_idx]
    confidence = predictions[0][pred_class_idx]
    
    logger.info("Confidence scores:")
    for idx, label in labels.items():
        logger.info(f"{label}: {predictions[0][idx]:.4f}")
    
    return pred_class, confidence, predictions[0]

def display_image(img, pred_class, confidence, image_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {pred_class} ({confidence:.4f})\n{os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()

def main():
    logger.info("Verifying model file...")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    logger.info("Loading trained model...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise
    
    logger.info(f"Selecting random images from {pred_path}...")
    image_files = [f for f in os.listdir(pred_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        logger.error(f"No images found in {pred_path}")
        raise FileNotFoundError(f"No images found in {pred_path}")
    
    # Select 5 random images
    selected_images = random.sample(image_files, min(5, len(image_files)))
    
    for img_file in selected_images:
        image_path = os.path.join(pred_path, img_file)
        img, img_array = load_and_preprocess_image(image_path)
        pred_class, confidence, scores = predict_image(model, img_array)
        logger.info(f"Predicted class: {pred_class} (Confidence: {confidence:.4f}) for {img_file}")
        display_image(img, pred_class, confidence, image_path)

if __name__ == "__main__":
    main()
