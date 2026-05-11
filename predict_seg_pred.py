
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

base_path = r"C:\Users\USER\Documents\main_cleaned"
pred_path = os.path.join(base_path, "seg_pred", "seg_pred")
model_path = r"C:\Users\USER\Documents\main\best_model.h5"
output_csv = os.path.join(base_path, "predictions.csv")

def verify_pred_dataset():
    logger.info("Verifying seg_pred dataset...")
    if os.path.exists(pred_path):
        image_files = [f for f in os.listdir(pred_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        logger.info(f"{pred_path}: {len(image_files)} images")
        if len(image_files) == 0:
            logger.error(f"No valid image files found in {pred_path}")
            raise FileNotFoundError(f"No valid image files found in {pred_path}")
    else:
        logger.error(f"{pred_path} does not exist!")
        raise FileNotFoundError(f"{pred_path} not found")
    
    logger.info("Verifying model file...")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    logger.info("seg_pred and model verification complete.")

def create_pred_generator():
    img_size = (224, 224)
    batch_size = 32

    pred_datagen = ImageDataGenerator(rescale=1./255)

    # Use a parent directory to treat seg_pred as a single class
    pred_generator = pred_datagen.flow_from_directory(
        os.path.join(base_path, "seg_pred"),  # Parent directory
        target_size=img_size,
        batch_size=batch_size,
        class_mode=None,  # No labels
        shuffle=False,    # Keep filenames in order
        classes=['seg_pred']  # Treat seg_pred as a single class
    )

    return pred_generator

def make_predictions(model, pred_generator):
    logger.info("Making predictions on seg_pred...")
    predictions = model.predict(pred_generator)
    
    labels = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}
    pred_classes = tf.argmax(predictions, axis=1)
    pred_labels = [labels[idx.numpy()] for idx in pred_classes]
    
    pred_filenames = pred_generator.filenames
    
    results = pd.DataFrame({
        'filename': pred_filenames,
        'predicted_class': pred_labels
    })
    
    results.to_csv(output_csv, index=False)
    logger.info(f"Predictions saved to {output_csv}")
    
    class_counts = results['predicted_class'].value_counts()
    logger.info("Predicted class distribution:")
    logger.info(class_counts)

def main():
    verify_pred_dataset()
    logger.info("Loading trained model...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise
    pred_generator = create_pred_generator()
    make_predictions(model, pred_generator)
    logger.info("Prediction completed!")

if __name__ == "__main__":
    main()
