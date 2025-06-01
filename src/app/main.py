import cv2
import time
from collections import deque, Counter
import numpy as np
import tensorflow as tf

class SignLanguageModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        # Replace with your actual label list
        self.labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")  # Add space as needed
    
    def predict(self, img):
        # Preprocess input image to model format here
        # For example resize, normalize, expand dims:
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        pred = self.model.predict(img)
        class_idx = np.argmax(pred)
        confidence = pred[0][class_idx]
        return self.labels[class_idx], confidence


def main():
    model_path = 'path/to/your/model'  # Update path accordingly
    model = SignLanguageModel(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    prediction_window = deque(maxlen=10)
    cooldown = 1.0  # seconds
    last_update_time = 0
    current_word = ""
    signing = False

    print("Press 's' to start signing, 'r' to reset word, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Define ROI box (example center square)
        h, w, _ = frame.shape
        box_size = 200
        x1 = w//2 - box_size//2
        y1 = h//2 - box_size//2
        x2 = x1 + box_size
        y2 = y1 + box_size
        roi = frame[y1:y2, x1:x2]

        # Draw ROI rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        if signing:
            pred_class, confidence = model.predict(roi)

            if confidence > 0.8:
                prediction_window.append(pred_class)
            else:
                prediction_window.append(None)

            if len(prediction_window) == prediction_window.maxlen:
                filtered_preds = [p for p in prediction_window if p is not None]
                if filtered_preds:
                    most_common_pred, count = Counter(filtered_preds).most_common(1)[0]
                    now = time.time()
                    if now - last_update_time > cooldown:
                        # Append to current word if not space
                        if most_common_pred == " ":
                            current_word += " "
                        else:
                            current_word += most_common_pred
                        last_update_time = now
                        print(f"Current word: {current_word}")

            # Show predicted letter on frame
            cv2.putText(frame, f"Letter: {pred_class} ({confidence:.2f})", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(frame, f"Word: {current_word}", (10, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("Sign Language Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            signing = True
            print("🟢 Started signing.")
            prediction_window.clear()
        elif key == ord('r'):
            current_word = ""
            prediction_window.clear()
            print("🔄 Word reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
