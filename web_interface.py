from flask import Flask, render_template, jsonify, request
from llm.autonomous_llm import AutonomousLLM
from llm.tts_module import TextToSpeech
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

app = Flask(__name__)

# Global variables
MODEL_PATH = "F:/lm-studio/models/QuantFactory/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-GGUF/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q8_0.gguf"
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
            tts = TextToSpeech()
            is_running = True
            
            # Get GPU info if available
            gpu_info = {
                "device": model.device,
                "memory_allocated": f"{torch.cuda.memory_allocated()/1024**3:.2f}GB" if torch.cuda.is_available() else "N/A",
                "memory_reserved": f"{torch.cuda.memory_reserved()/1024**3:.2f}GB" if torch.cuda.is_available() else "N/A"
            }
            
            logger.info(f"System started successfully on {gpu_info['device']}")
            return jsonify({
                "status": "started",
                "gpu_info": gpu_info
            })
        return jsonify({"status": "already running"})
    except Exception as e:
        logger.error(f"Error starting system: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stop', methods=['POST'])
def stop_system():
    global model, tts, is_running
    try:
        if is_running:
            logger.info("Stopping system...")
            # Clear the message queue
            while not message_queue.empty():
                message_queue.get_nowait()
                
            is_running = False
            if model:
                model.close_model()
                model = None
            if tts:
                tts = None
                
            # Add shutdown message to queue
            message_queue.put({
                "type": "system",
                "content": "System stopped",
                "timestamp": time.strftime("%H:%M:%S")
            })
            
            logger.info("System stopped successfully")
            return jsonify({"status": "stopped"})
        return jsonify({"status": "already stopped"})
    except Exception as e:
        logger.error(f"Error stopping system: {e}")
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
def submit_prompt():
    global model, tts, is_running
    
    if not is_running:
        return jsonify({"error": "System not running"}), 400
    
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        # Get relevant memories
        relevant_memories = get_relevant_memories(prompt, memories)
        
        # Build context with memories
        context = ""
        if relevant_memories:
            context += "Here are some relevant memories you should consider:\n"
            for memory in relevant_memories:
                context += f"- {memory['content']} (from {memory['timestamp']})\n"
        
        context += "\nBased on these memories and the current prompt, please respond appropriately.\n"
        
        # Combine context and prompt
        full_prompt = f"{context}\nCurrent prompt: {prompt}"
        
        # Process with model
        response = model.process_prompt(full_prompt)
        
        # Check for memory-related keywords in prompt or response
        memory_keywords = ['remember', 'recall', 'memory', 'forget', 'store']
        should_store = any(keyword in prompt.lower() for keyword in memory_keywords) or \
                      any(keyword in response.lower() for keyword in memory_keywords)
        
        if should_store:
            new_memory = {
                'content': f"Prompt: {prompt}\nResponse: {response}",
                'timestamp': datetime.now().isoformat(),
                'type': 'conversation'
            }
            memories.append(new_memory)
            save_memories()  # Make sure this function exists to persist memories
        
        return jsonify({
            "response": response,
            "memories_used": len(relevant_memories),
            "tokens": model.get_token_count()  # Make sure this method exists
        })
        
    except Exception as e:
        logger.error(f"Error processing prompt: {str(e)}", exc_info=True)
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)