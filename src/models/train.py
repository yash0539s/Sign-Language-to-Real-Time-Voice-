import os
import yaml
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import mlflow
import mlflow.tensorflow
from src.preprocessing.data_preocessing import get_data_generators


def build_model(input_shape, num_classes):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base layers initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    # Load configuration
    config_path = os.path.join("configs", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"⚠️ Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract config parameters
    train_dir = config['dataset']['train_path']
    test_dir = config['dataset']['test_path']
    image_size = tuple(config['dataset']['image_size'])
    batch_size = config['dataset']['batch_size']
    input_shape = tuple(config['model']['input_shape'])
    num_classes = config['model']['num_classes']
    epochs = config['model']['epochs']

    # Setup MLflow tracking
    tracking_uri = config['mlflow'].get('tracking_uri', None)
    if not tracking_uri:
        tracking_uri = "file://" + os.path.abspath("mlruns")
        print(f"⚠️ No MLflow tracking_uri set in config, using local uri: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # Load data
    print("📦 Loading data...")
    train_dataset, test_dataset = get_data_generators(train_dir, test_dir, image_size, batch_size)

    # Build model
    model = build_model(input_shape, num_classes)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)

    # Start MLflow run
    with mlflow.start_run():
        print("🚀 Starting model training...")
        history = model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=epochs,
            callbacks=[early_stopping, model_checkpoint]
        )
        print("✅ Training complete. Logging to MLflow...")

        # Log parameters
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "input_shape": input_shape,
            "num_classes": num_classes,
            "model_type": "MobileNetV2"
        })

        # Log final metrics
        mlflow.log_metrics({
            "train_accuracy": history.history['accuracy'][-1],
            "val_accuracy": history.history['val_accuracy'][-1],
            "train_loss": history.history['loss'][-1],
            "val_loss": history.history['val_loss'][-1]
        })

        # Save and log model
        model_dir = os.path.join("models", config['model']['name'])
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "saved_model2.keras")  # use .keras extension
        model.save(model_path)
        print(f"✅ Model saved locally at '{model_path}'")

        mlflow.keras.log_model(model, artifact_path="model")
        print("✅ Model logged to MLflow under artifact path 'model'")

    print("🎉 Training and logging complete!")


if __name__ == "__main__":
    main()
