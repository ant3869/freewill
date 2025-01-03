# File: llm/wake_word.py
# Purpose: Detects a wake word using speech recognition with error handling and threading.

import speech_recognition as sr
import numpy as np
import random
from scipy.signal import lfilter


class WakeWordDetector:
    def __init__(self, wake_word="assistant"):
        """Initialize the Wake Word Detector."""
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.wake_word = wake_word.lower()

    def _filter_audio(self, audio_data):
        """Apply basic noise reduction using a low-pass filter."""
        return lfilter([1], [1, -0.95], audio_data)

    def listen_for_wake_word(self, command_queue):
        """Continuously listen for the wake word."""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("[WakeWordDetector] Listening for wake word...")
            while True:
                try:
                    audio = self.recognizer.listen(source)
                    detected_text = self.recognizer.recognize_google(audio).lower()
                    print(f"[WakeWordDetector] Detected: {detected_text}")
                    if self.wake_word in detected_text:
                        print("[WakeWordDetector] Wake word detected!")
                        command_queue.put("wake")
                except sr.UnknownValueError:
                    # Wake word not detected; continue listening
                    continue
                except sr.RequestError as e:
                    print(f"[WakeWordDetector] Speech recognition error: {e}")
                    break

    def detect(self):
    # Simulate wake-word detection (random for now)
        return random.choice([True, False])