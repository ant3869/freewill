import pyttsx3
import logging
import threading

# Set up logger
logger = logging.getLogger(__name__)

class TTSModule:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.is_speaking = False
        self._lock = threading.Lock()
        
        # Default settings - explicitly enabled
        self.settings = {
            'internal_enabled': True,  # Default to ON
            'external_enabled': True,  # Default to ON
            'rate': 150,
            'volume': 1.0,
            'voice': None
        }
        
        # Apply default settings immediately
        self.apply_settings()
        logger.info("TTS initialized with default settings enabled")

    def apply_settings(self):
        try:
            self.engine.setProperty('rate', self.settings['rate'])
            self.engine.setProperty('volume', self.settings['volume'])
            if self.settings['voice']:
                self.engine.setProperty('voice', self.settings['voice'])
            logger.info(f"TTS settings applied: {self.settings}")
        except Exception as e:
            logger.error(f"Error applying TTS settings: {e}")

    def update_settings(self, new_settings):
        try:
            self.settings.update(new_settings)
            self.apply_settings()
            logger.info("TTS settings updated successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating TTS settings: {e}")
            return False

    def should_speak(self, is_internal=False):
        """Check if TTS should be used based on message type"""
        return (self.settings['internal_enabled'] if is_internal 
                else self.settings['external_enabled'])

    def speak(self, text, is_internal=False):
        """Speak text if appropriate setting is enabled"""
        if not self.should_speak(is_internal):
            return False
            
        try:
            with self._lock:
                self.is_speaking = True
                self.engine.say(text)
                self.engine.runAndWait()
                self.is_speaking = False
            return True
        except Exception as e:
            logger.error(f"TTS error: {e}")
            self.is_speaking = False
            return False