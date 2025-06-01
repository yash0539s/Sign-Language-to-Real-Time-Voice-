import os
import sys
import yaml
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(train_dir, val_dir, image_size, batch_size):
    # Check if train and validation directories exist
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"❌ Error: One or both of the following directories do not exist:\n - {train_dir}\n - {val_dir}")
        sys.exit(1)

    # Augmentation for training data
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.15,
    fill_mode='nearest'
)
    # Only rescale for validation data
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Train generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Print information
    print(f"\n🔠 Classes found: {train_generator.class_indices}")
    print(f"📸 Total training images: {train_generator.samples}")
    print(f"📸 Total validation images: {val_generator.samples}")

    return train_generator, val_generator

if __name__ == "__main__":
    # Load config.yaml
    config_path = os.path.join("configs", "config.yaml")
    if not os.path.exists(config_path):
        print("❌ Error: config.yaml not found in /configs folder.")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract configuration
    train_dir = config['dataset']['train_path']
    val_dir = config['dataset']['test_path']  # assuming test_path is your val_dir
    image_size = tuple(config['dataset']['image_size'])
    batch_size = config['dataset'].get('batch_size', 8)  # fallback default

    print("📦 Starting data preprocessing...")

    # Run preprocessing
    get_data_generators(train_dir, val_dir, image_size, batch_size)

    print("✅ Preprocessing complete.")
