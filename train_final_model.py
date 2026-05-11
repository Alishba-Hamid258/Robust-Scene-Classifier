
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2S
import logging
from config import TRAIN_DIR, TEST_DIR, BEST_MODEL_PATH

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CONFIGURATION
TRAIN_DIR = TRAIN_DIR
TEST_DIR = TEST_DIR
MODEL_SAVE_PATH = BEST_MODEL_PATH

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 15

def load_datasets():
    logger.info("Loading datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )
    
    # Prefetch for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds

def build_model(num_classes):
    logger.info("Building model with EfficientNetV2S...")
    
    # Data Augmentation Layers
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
    ])
    
    base_model = EfficientNetV2S(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False  # Freeze initially
    
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        data_augmentation,
        layers.Rescaling(1./255),  # Normalization
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

def train():
    train_ds, val_ds = load_datasets()
    num_classes = 7
    
    model, base_model = build_model(num_classes)
    
    # PHASE 1: Training the top layers
    logger.info("Phase 1: Training top layers...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
    ]
    
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE1, callbacks=callbacks)
    
    # PHASE 2: Fine-tuning
    logger.info("Phase 2: Fine-tuning the entire model...")
    base_model.trainable = True
    
    # Recompile with a very low learning rate and Cosine Decay
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-5,
        decay_steps=len(train_ds) * EPOCHS_PHASE2
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE2, callbacks=callbacks)
    
    logger.info(f"Training complete. Best model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
