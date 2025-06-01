# src/utils/speech.py

import pyttsx3

class Speech:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 150)  # Speed

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()
