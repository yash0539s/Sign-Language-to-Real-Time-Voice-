import tensorflow as tf
import numpy as np
import cv2

class SignLanguageModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        # Your dataset has only letters A-Z
        self.labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def preprocess(self, img):
        img = cv2.resize(img, (64, 64))  # resize to your model input size
        img = img / 255.0
        return np.expand_dims(img, axis=0)

    def predict(self, img):
        processed = self.preprocess(img)
        preds = self.model.predict(processed)
        class_idx = np.argmax(preds)
        confidence = np.max(preds)
        # Sanity check in case model outputs invalid index
        if class_idx >= len(self.labels):
            return None, 0.0
        return self.labels[class_idx], confidence
