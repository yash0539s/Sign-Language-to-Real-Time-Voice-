from src.models.predict import SignLanguageModel
from src.models.speech import Speech
import cv2

def main():
    model_path = "models/asl_cnn_model_v2/saved_model2.keras"
    model = SignLanguageModel(model_path)
    speech = Speech()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    word = ""
    print("Press 's' to capture letter, 'v' to speak word, 'r' to reset, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        x1, y1, x2, y2 = 100, 100, 300, 300
        roi = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            prediction = model.predict(roi)
            word += prediction
            print(f"Captured letter: {prediction}")

        elif key == ord('v'):
            if word:
                print(f"Full word: {word}")
                speech.speak(word)
                word = ""  # reset after speaking
            else:
                print("No word to speak.")

        elif key == ord('r'):
            word = ""
            print("Word reset.")

        elif key == ord('q'):
            break

        cv2.putText(frame, f"Word: {word}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Sign Language Recognition", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
