import pyttsx3
import logging

class TextToSpeech:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.engine = pyttsx3.init()
            
            # Configure the engine
            self.engine.setProperty('rate', 150)    # Speed of speech
            self.engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
            
            # Get available voices and set a good one
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to find a female voice
                female_voice = next((voice for voice in voices if 'female' in voice.name.lower()), None)
                if female_voice:
                    self.engine.setProperty('voice', female_voice.id)
                else:
                    # Use the first available voice
                    self.engine.setProperty('voice', voices[0].id)
            
            self.logger.info("TTS engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize TTS engine: {str(e)}")
            raise

    def speak(self, text):
        """Speak the given text"""
        try:
            if not text:
                self.logger.warning("Empty text provided to TTS")
                return
                
            self.logger.info(f"Speaking text (length: {len(text)})")
            self.engine.say(text)
            self.engine.runAndWait()
            self.logger.info("Finished speaking")
        except Exception as e:
            self.logger.error(f"Error in TTS speak: {str(e)}")
            raise

    def stop(self):
        """Stop any ongoing speech and clean up"""
        try:
            self.engine.stop()
            self.logger.info("TTS stopped")
        except Exception as e:
            self.logger.error(f"Error stopping TTS: {str(e)}")

# # File: llm/tts_module.py
# # Purpose: Provides a wrapper for offline Text-to-Speech synthesis using Coqui TTS.

# import os
# import subprocess
# import threading
# from TTS.api import TTS


# class TextToSpeech:
#     def __init__(self, tts_model="tts_models/en/ljspeech/tacotron2-DDC"):
#         """Initialize TTS with the specified model."""
#         self.tts = TTS(tts_model)
#         self.lock = threading.Lock()  # Prevent overlapping TTS calls

#     def speak(self, text, output_path="output.wav"):
#         """Generate speech from text and play it."""
#         with self.lock:  # Ensure only one TTS playback at a time
#             print("[TTS] Generating speech...")
#             self.tts.tts_to_file(text=text, file_path=output_path)
#             self._play_audio(output_path)

#     def _play_audio(self, file_path):
#         """Play the generated audio file and wait for completion."""
#         try:
#             if os.name == "posix":  # Linux/Mac
#                 subprocess.run(["aplay", file_path], check=True)
#             elif os.name == "nt":  # Windows
#                 subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{file_path}').PlaySync()"], check=True)
#             else:
#                 raise NotImplementedError("Audio playback not supported for this OS.")
#         except subprocess.CalledProcessError as e:
#             print(f"[TTS] Error playing audio: {e}")