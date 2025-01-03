from flask import Flask, render_template, jsonify, request
from llm.autonomous_llm import AutonomousLLM
from llm.tts_module import TTSModule
import torch
import psutil
import GPUtil
from threading import Thread
import time
import queue
import time
import logging
import os
from logging.handlers import RotatingFileHandler
import pyttsx3
import threading
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import partial
import atexit

app = Flask(__name__)

# Global variables
MODEL_PATH = "models/DarkIdol-Llama-3_1.gguf"
model = None
tts = None
is_running = False
message_queue = queue.Queue()
memories = []

# Set up logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = RotatingFileHandler(
    'logs/web_interface.log',
    maxBytes=1024 * 1024,
    backupCount=5
)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Global variables for system stats
system_stats = {
    'cpu': 0,
    'ram': 0,
    'gpu': None
}

# Create thread pool for async operations
executor = ThreadPoolExecutor(max_workers=3)

# Add to your global variables
system_prompts = {
    'main': '',
    'internal': '',
    'external': ''
}

def update_system_stats():
    """Background thread to update system statistics"""
    global system_stats
    while True:
        try:
            # CPU Usage
            system_stats['cpu'] = psutil.cpu_percent(interval=1)
            
            # RAM Usage
            memory = psutil.virtual_memory()
            system_stats['ram'] = memory.percent
            
            # GPU Usage (if available)
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Get first GPU
                    system_stats['gpu'] = {
                        'name': gpu.name,
                        'load': gpu.load * 100,
                        'memory': {
                            'used': gpu.memoryUsed,
                            'total': gpu.memoryTotal
                        }
                    }
            except Exception as e:
                logger.error(f"Error getting GPU stats: {e}")
                system_stats['gpu'] = None
                
        except Exception as e:
            logger.error(f"Error updating system stats: {e}")
        
        time.sleep(1)  # Update every second

# Start the monitoring thread
stats_thread = Thread(target=update_system_stats, daemon=True)
stats_thread.start()

# Add new route for system stats
@app.route('/system_stats')
def get_system_stats():
    return jsonify(system_stats)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_system():
    global model, tts, is_running
    try:
        if not is_running:
            logger.info("Starting system...")
            model = AutonomousLLM(MODEL_PATH)
            tts = TTSModule()
            is_running = True
            return jsonify({"status": "started"})
        return jsonify({"status": "already running"})
    except Exception as e:
        logger.error(f"Error starting system: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stop', methods=['POST'])
async def stop_system():
    try:
        logger.info("Received stop request")
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, model.close_model)
        
        logger.info("System stopped successfully")
        return jsonify({"status": "stopped"})
        
    except Exception as e:
        logger.error(f"Error stopping system: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def get_relevant_memories(prompt, memories, max_memories=5):
    """Get memories relevant to the current prompt"""
    relevant_memories = []
    
    # Convert prompt to lowercase for matching
    prompt_lower = prompt.lower()
    
    for memory in memories:
        # Check for exact keyword matches first
        if any(word in memory['content'].lower() for word in prompt_lower.split()):
            relevant_memories.append(memory)
            continue
            
        # Then check for semantic similarity (if we had embedding comparison)
        # For now, using simple substring matching
        if any(part in memory['content'].lower() for part in prompt_lower.split()):
            relevant_memories.append(memory)
    
    # Sort by timestamp (newest first) and limit
    relevant_memories.sort(key=lambda x: x['timestamp'], reverse=True)
    return relevant_memories[:max_memories]

@app.route('/submit', methods=['POST'])
async def submit():
    try:
        data = request.get_json()
        logger.info(f"Received submit request: {data}")
        
        # Handle prompt processing in separate thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            partial(process_prompt, data.get('prompt', ''))
        )
        
        logger.info(f"Processed prompt with result: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in submit: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def save_memories():
    """Save memories to persistent storage"""
    try:
        with open('memories.json', 'w') as f:
            json.dump(memories, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving memories: {str(e)}")

def load_memories():
    """Load memories from persistent storage"""
    global memories
    try:
        if os.path.exists('memories.json'):
            with open('memories.json', 'r') as f:
                memories = json.load(f)
        else:
            memories = []
    except Exception as e:
        logger.error(f"Error loading memories: {str(e)}")
        memories = []

# Initialize memories on startup
load_memories()

@app.route('/get_messages')
def get_messages():
    messages = []
    try:
        while not message_queue.empty():
            messages.append(message_queue.get_nowait())
        return jsonify(messages)
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_logs')
def get_logs():
    try:
        with open('logs/web_interface.log', 'r') as f:
            # Get last 100 lines of logs
            lines = f.readlines()[-100:]
            return jsonify({"logs": lines})
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    try:
        logger.info("Attempting to clear logs")
        log_file = 'logs/web_interface.log'
        
        # First try to clear the file
        open(log_file, 'w').close()
        
        # Add a new entry indicating logs were cleared
        logger.info("=== Logs cleared ===")
        
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Error clearing logs: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/cancel', methods=['POST'])
def cancel_generation():
    global model
    try:
        if model:
            # Implement model-specific cancellation
            model.stop_generation()  # You'll need to implement this in your AutonomousLLM class
            logger.info("Generation cancelled by user")
            return jsonify({"status": "cancelled"})
        return jsonify({"status": "no active generation"})
    except Exception as e:
        logger.error(f"Error cancelling generation: {e}")
        return jsonify({"error": str(e)}), 500

class TTSHandler:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.is_speaking = False
        self._lock = threading.Lock()

    def speak(self, text):
        try:
            with self._lock:
                if self.is_speaking:
                    self.engine.stop()
                self.is_speaking = True
                
            # Reinitialize engine for each new speech
            self.engine = pyttsx3.init()
            self.engine.say(text)
            self.engine.runAndWait()
            
        except Exception as e:
            logger.error(f"TTS Error: {str(e)}")
        finally:
            with self._lock:
                self.is_speaking = False
                try:
                    self.engine.stop()
                except:
                    pass

    def stop(self):
        with self._lock:
            if self.is_speaking:
                try:
                    self.engine.stop()
                except:
                    pass
                self.is_speaking = False

@app.route('/select_model_folder', methods=['GET'])
def select_model_folder():
    try:
        # Default model path
        default_model = 'DarkIdol-Llama-3_1.gguf'
        models = [default_model]  # Start with default model
        
        # Add any other models found in the directory
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        if os.path.exists(model_dir):
            models.extend([f for f in os.listdir(model_dir) 
                         if f.endswith('.gguf') and f != default_model])
        
        return jsonify({"models": models})
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/update_settings', methods=['POST'])
async def update_settings():
    try:
        settings = request.get_json()
        logger.info(f"Updating settings: {settings}")
        
        # Update model settings in separate thread
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor,
            partial(model.update_settings, settings)
        )
        
        logger.info("Settings updated successfully")
        return jsonify({"status": "success"})
        
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Add cleanup on shutdown
@atexit.register
def cleanup():
    executor.shutdown(wait=False)

def process_prompt(prompt):
    """Process the prompt using the loaded model"""
    try:
        if not model:
            raise Exception("Model not loaded")
        
        logger.info(f"Processing prompt: {prompt}")
        # Remove system_prompt parameter
        response = model.generate_response(prompt)
        return {"response": response}
        
    except Exception as e:
        logger.error(f"Error processing prompt: {e}")
        raise

@app.route('/update_prompts', methods=['POST'])
async def update_prompts():
    global system_prompts
    try:
        prompts = request.get_json()
        logger.info(f"Updating system prompts: {prompts}")
        system_prompts.update(prompts)
        
        if model:  # If model is loaded, update it
            await asyncio.get_event_loop().run_in_executor(
                executor,
                partial(model.update_prompts, **system_prompts)
            )
        
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Error updating prompts: {e}")
        return jsonify({"error": str(e)}), 500

class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.is_speaking = False
        self._lock = threading.Lock()
        # Default settings
        self.internal_tts_enabled = True
        self.external_tts_enabled = True
        self.rate = 150  # Default speech rate
        self.volume = 1.0
        self.voice = None
        self.apply_settings()

    def apply_settings(self):
        try:
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            if self.voice:
                self.engine.setProperty('voice', self.voice)
        except Exception as e:
            logger.error(f"Error applying TTS settings: {e}")

    def update_settings(self, settings):
        try:
            if 'rate' in settings:
                self.rate = settings['rate']
            if 'volume' in settings:
                self.volume = settings['volume']
            if 'voice' in settings:
                self.voice = settings['voice']
            if 'internal_enabled' in settings:
                self.internal_tts_enabled = settings['internal_enabled']
            if 'external_enabled' in settings:
                self.external_tts_enabled = settings['external_enabled']
            
            self.apply_settings()
            logger.info("TTS settings updated successfully")
        except Exception as e:
            logger.error(f"Error updating TTS settings: {e}")
            raise

if __name__ == '__main__':
    app.run(debug=True, port=5000)