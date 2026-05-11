
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
try:
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError(f"Required library missing: {str(e)}. Install with 'pip install scikit-learn seaborn matplotlib'")
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

base_path = r"C:\Users\USER\Documents\main_cleaned"
test_path = os.path.join(base_path, "seg_test", "seg_test")
model_path = r"C:\Users\USER\Documents\main\best_model.h5"
output_dir = base_path

def verify_test_dataset():
    logger.info("Verifying seg_test dataset...")
    if os.path.exists(test_path):
        subfolders = [f for f in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, f))]
        logger.info(f"{test_path}: {len(subfolders)} subfolders")
        if len(subfolders) != 6:
            logger.warning(f"Expected 6 subfolders, found {len(subfolders)}")
    else:
        logger.error(f"{test_path} does not exist!")
        raise FileNotFoundError(f"{test_path} not found")
    
    logger.info("Verifying model file...")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    logger.info("seg_test and model verification complete.")

def create_test_generator():
    img_size = (224, 224)
    batch_size = 32

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return test_generator

def evaluate_model(model, test_generator):
    logger.info("Evaluating model on seg_test...")
    predictions = model.predict(test_generator)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    labels = {v: k for k, v in test_generator.class_indices.items()}
    class_names = [labels[i] for i in range(len(labels))]
    
    # Compute metrics
    report = classification_report(true_classes, pred_classes, target_names=class_names, output_dict=True)
    cm = confusion_matrix(true_classes, pred_classes)
    
    # Log metrics
    logger.info("Classification Report:")
    for class_name in class_names:
        logger.info(f"{class_name}: Precision={report[class_name]['precision']:.4f}, "
                    f"Recall={report[class_name]['recall']:.4f}, "
                    f"F1-Score={report[class_name]['f1-score']:.4f}")
    logger.info(f"Overall Accuracy: {report['accuracy']:.4f}")
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.show()
    logger.info(f"Confusion matrix saved to {cm_path}")

def main():
    verify_test_dataset()
    logger.info("Loading trained model...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise
    test_generator = create_test_generator()
    evaluate_model(model, test_generator)
    logger.info("Model evaluation completed!")

if __name__ == "__main__":
    main()
