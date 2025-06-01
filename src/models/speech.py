# src/utils/speech.py

import pyttsx3

class Speech:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()
