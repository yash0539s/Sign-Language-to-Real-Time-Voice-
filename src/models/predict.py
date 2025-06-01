# src/models/predict.py

import tensorflow as tf
import numpy as np
import cv2

class SignLanguageModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.labels = [chr(i) for i in range(65, 91)]  # A-Z

    def predict(self, image):
        image = cv2.resize(image, (64, 64))
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)
        preds = self.model.predict(image)
        return self.labels[np.argmax(preds)]
