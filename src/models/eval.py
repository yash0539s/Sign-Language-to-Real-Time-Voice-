import os
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from src.preprocessing.data_preocessing import get_data_generators

def load_config():
    config_path = os.path.join("configs", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def main():
    print("🔍 Starting evaluation...")

    config = load_config()

    test_dir = config['dataset']['test_path']
    image_size = tuple(config['dataset']['image_size'])
    batch_size = config['dataset']['batch_size']
    model_name = config['model']['name']
    model_path = os.path.join("models", model_name, "saved_model2.keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Trained model not found: {model_path}")

    print("📦 Loading test data...")
    _, test_generator = get_data_generators(test_dir, test_dir, image_size, batch_size)
    class_names = list(test_generator.class_indices.keys())

    print(f"🧠 Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print("🧪 Evaluating model...")
    loss, acc = model.evaluate(test_generator, verbose=1)
    print(f"\n✅ Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

    # Predict
    print("🔮 Predicting classes...")
    y_true = test_generator.classes
    y_pred_probs = model.predict(test_generator, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification Report
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    print("📈 Plotting confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, class_names)

if __name__ == "__main__":
    main()
