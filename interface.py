from flask import Flask, render_template, jsonify, request
from llm.autonomous_llm import AutonomousLLM
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
import psutil
import GPUtil
import asyncio
import logging
import os
from flask import jsonify
import json
import queue
import time
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import atexit
from llm.tts_module import TTSModule
from flask import Flask, jsonify

app = Flask(__name__)
LOG_FILE = "logs/web_interface.log"

@app.route('/select_model_folder')
def select_model_folder():
    try:
        model_dir = "models"  # Local models directory
        print(f"Scanning directory: {model_dir}")
        models = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]
        print(f"Found models: {models}")
        return jsonify({"models": models})
    except Exception as e:
        print(f"Error scanning model directory: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_logs')
def get_logs():
    try:
        with open('logs/web_interface.log', 'r') as f:
            logs = f.readlines()[-100:]  # Get last 100 lines
            formatted_logs = []
            for line in logs:
                parts = line.split(' - ')
                if len(parts) >= 3:
                    timestamp = parts[0]
                    level = parts[1].strip()
                    message = ' - '.join(parts[2:]).strip()
                    formatted_logs.append({
                        'timestamp': timestamp,
                        'level': level.lower(),
                        'message': message
                    })
            return jsonify({'logs': formatted_logs})
    except FileNotFoundError:
        return jsonify({'error': 'Log file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    


# Configuration
@dataclass
class ServerConfig:
    model_path: str
    models_directory: str = "models"  # Default models directory
    max_workers: int = 3
    log_path: str = 'logs'
    log_max_bytes: int = 1024 * 1024
    log_backup_count: int = 5
    tts_enabled: bool = True

    def __post_init__(self):
        # Ensure models directory is absolute
        if not os.path.isabs(self.models_directory):
            self.models_directory = os.path.abspath(self.models_directory)
        
        # If model_path is just a filename, assume it's in models directory
        if not os.path.isabs(self.model_path):
            self.model_path = os.path.join(self.models_directory, self.model_path)

class WebInterface:
    def __init__(self, config: ServerConfig):
        self.app = Flask(__name__)
        self.config = config
        self.model: Optional[AutonomousLLM] = None
        self.is_running = False
        self.message_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.setup_logging()
        self.setup_routes()
        self.system_stats = {'cpu': 0, 'ram': 0, 'gpu': None}
        self.tts = None if not config.tts_enabled else TTSModule()
        self.memories = []
        self.system_prompts = {
            'main': '',
            'internal': '',
            'external': ''
        }
        
        # Register cleanup
        atexit.register(self.cleanup)

    def setup_logging(self) -> None:
        """Configure logging with rotation and formatting."""
        if not os.path.exists(self.config.log_path):
            os.makedirs(self.config.log_path)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        handler = RotatingFileHandler(
            os.path.join('logs', 'web_interface.log'),
            maxBytes=self.config.log_max_bytes,
            backupCount=self.config.log_backup_count
        )
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def setup_routes(self) -> None:
        """Set up Flask routes with proper error handling."""
        self.app.route('/')(self.home)
        self.app.route('/start', methods=['POST'])(self.start_system)
        self.app.route('/stop', methods=['POST'])(self.stop_system)
        self.app.route('/submit', methods=['POST'])(self.submit)
        self.app.route('/system_stats')(self.get_system_stats)
        self.app.route('/get_messages')(self.get_messages)
        self.app.route('/get_logs')(self.get_logs)
        self.app.route('/clear_logs', methods=['POST'])(self.clear_logs)
        self.app.route('/update_settings', methods=['POST'])(self.update_settings)
        self.app.route('/update_prompts', methods=['POST'])(self.update_prompts)
        self.app.route('/cancel', methods=['POST'])(self.cancel_generation)
        self.app.route('/select_model_folder')(self.select_model_folder)
        #self.app.route('/select_model_folder', methods=['GET'])(self.select_model_folder)

    def home(self):
        """Render the main interface."""
        return render_template('index.html')

    async def start_system(self):
        """Start the LLM system with proper error handling."""
        try:
            if not self.is_running:
                self.logger.info("Starting system...")
                
                # Get model path from request
                data = request.get_json()
                model_name = data.get('model')
                
                if not model_name:
                    model_name = "DarkIdol-Llama-3_1.gguf"
                
                # Find full path for model
                model_path = "C:/Users/antho/Desktop/stuff/projects/freewill/models/DarkIdol-Llama-3_1.gguf"
                if os.path.isabs(model_name):
                    model_path = model_name
                else:
                    # Search in configured directories
                    model_dirs = [
                        self.config.models_directory,
                        "models",
                        os.path.expanduser("~/models")
                    ]
                    
                    for directory in model_dirs:
                        potential_path = os.path.join(directory, model_name)
                        if os.path.exists(potential_path):
                            model_path = potential_path
                            break
                
                if not model_path:
                    raise FileNotFoundError(f"Model {model_name} not found in configured directories")
                
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    self.executor,
                    lambda: AutonomousLLM(model_path)
                )
                
                self.is_running = True
                self.logger.info(f"System started successfully with model: {model_path}")
                return jsonify({"status": "started"})
            return jsonify({"status": "already running"})
            
        except Exception as e:
            self.logger.error(f"Error starting system: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    async def stop_system(self):
        """Stop the LLM system safely."""
        try:
            if self.is_running and self.model:
                self.logger.info("Stopping system...")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self.executor, self.model.cleanup)
                self.model = None
                self.is_running = False
                self.logger.info("System stopped successfully")
                return jsonify({"status": "stopped"})
            return jsonify({"status": "not running"})
            
        except Exception as e:
            self.logger.error(f"Error stopping system: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    async def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """Process a prompt through the LLM."""
        try:
            if not self.model:
                raise ValueError("Model not initialized")
                
            self.logger.info(f"Processing prompt: {prompt}")
            response = await self.model.generate_response(prompt)
            
            # Add to message queue
            self.message_queue.put({
                'content': response['response'],
                'timestamp': time.strftime('%H:%M:%S'),
                'type': 'external'
            })
            
            return {"response": response['response']}
            
        except Exception as e:
            self.logger.error(f"Error processing prompt: {str(e)}", exc_info=True)
            raise

        # @app.route('/submit', methods=['POST'])
        # async def submit(self):
        #     try:
        #         data = request.get_json()
        #         prompt = data.get('prompt')
        #         result = await self.model.generate(prompt)  # Add self.
        #         return jsonify(result)
        #     except Exception as e:
        #         self.logger.error(f"Error in submit: {str(e)}", exc_info=True)
        #         return jsonify({"error": str(e)}), 500

    @app.route('/submit', methods=['POST'])
    async def submit(self):
        try:
            data = request.get_json()
            prompt = data.get('prompt')
            result = await self.process_prompt(prompt)  # Only await once
            return jsonify({"response": result})
        except Exception as e:
            self.logger.error(f"Error in submit: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    def get_system_stats(self):
        """Get current system resource usage."""
        try:
            self.update_system_stats()
            return jsonify(self.system_stats)
        except Exception as e:
            self.logger.error(f"Error getting system stats: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def get_messages(self):
        """Get queued messages."""
        try:
            messages = []
            while not self.message_queue.empty():
                messages.append(self.message_queue.get_nowait())
            return jsonify(messages)
        except Exception as e:
            self.logger.error(f"Error getting messages: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def get_logs(self):
        try:
            with open('logs/web_interface.log', 'r') as f:
                logs = f.readlines()[-100:]  # Get last 100 lines
                formatted_logs = []
                for line in logs:
                    parts = line.split(' - ')
                    if len(parts) >= 3:
                        timestamp = parts[0]
                        level = parts[1].strip()
                        message = ' - '.join(parts[2:]).strip()
                        formatted_logs.append({
                            'timestamp': timestamp,
                            'level': level.lower(),
                            'message': message
                        })
                return jsonify({'logs': formatted_logs})
        except Exception as e:
            return jsonify({'logs': []})

    def clear_logs(self):
        """Clear log file contents."""
        try:
            log_file = os.path.join(self.config.log_path, 'web_interface.log')
            open(log_file, 'w').close()
            self.logger.info("Logs cleared")
            return jsonify({"status": "success"})
        except Exception as e:
            self.logger.error(f"Error clearing logs: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def update_settings(self):
        """Update system settings."""
        try:
            settings = request.get_json()
            if self.model:
                self.model.update_settings(settings)
            return jsonify({"status": "success"})
        except Exception as e:
            self.logger.error(f"Error updating settings: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def update_prompts(self):
        """Update system prompts."""
        try:
            prompts = request.get_json()
            self.system_prompts.update(prompts)
            if self.model:
                self.model.update_prompts(**self.system_prompts)
            return jsonify({"status": "success"})
        except Exception as e:
            self.logger.error(f"Error updating prompts: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def update_system_stats(self) -> None:
        """Update system resource usage stats."""
        try:
            self.system_stats['cpu'] = psutil.cpu_percent()
            self.system_stats['ram'] = psutil.virtual_memory().percent
            
            if torch.cuda.is_available():
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.system_stats['gpu'] = {
                        'name': gpu.name,
                        'load': gpu.load * 100,
                        'memory': {
                            'used': gpu.memoryUsed,
                            'total': gpu.memoryTotal
                        }
                    }
            
        except Exception as e:
            self.logger.error(f"Error updating system stats: {str(e)}")

    def cleanup(self) -> None:
        """Clean up resources on shutdown."""
        try:
            if self.model:
                self.model.cleanup()
            self.executor.shutdown(wait=False)
            self.logger.info("Cleanup completed successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application."""
        self.app.run(host=host, port=port, debug=debug)

    async def cancel_generation(self):
        try:
            if self.model:
                self.model.stop_generation()
                self.logger.info("Generation cancelled by user")
                return jsonify({"status": "cancelled"})
            return jsonify({"status": "no active generation"})
        except Exception as e:
            self.logger.error(f"Error cancelling generation: {e}")
            return jsonify({"error": str(e)}), 500

    def select_model_folder(self):
        """List available models in the models directory."""
        try:
            # Default model (can be configured in a config file)
            default_model = 'models/DarkIdol-Llama-3_1.gguf'
            models = [default_model]
            
            # Check both the models directory and any configured external directories
            model_dirs = [
                self.config.models_directory,
                "models",  # Add common LM Studio path
                os.path.expanduser("~/models"),  # User's home directory
            ]
            
            for model_dir in model_dirs:
                if os.path.exists(model_dir):
                    # Find all .gguf files (common LLaMA model format)
                    for root, _, files in os.walk(model_dir):
                        for file in files:
                            if file.endswith('.gguf'):
                                # Store full path but display only filename
                                full_path = os.path.join(root, file)
                                models.append({
                                    'name': file,
                                    'path': full_path
                                })
            
            self.logger.info(f"Found {len(models)} models")
            return jsonify({"models": models})
            
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return jsonify({"error": str(e)}), 500

    def save_memories(self):
        try:
            with open('memories.json', 'w') as f:
                json.dump(self.memories, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving memories: {str(e)}")

    def load_memories(self):
        try:
            if os.path.exists('memories.json'):
                with open('memories.json', 'r') as f:
                    self.memories = json.load(f)
            else:
                self.memories = []
        except Exception as e:
            self.logger.error(f"Error loading memories: {str(e)}")
            self.memories = []


if __name__ == '__main__':
    config = ServerConfig(
        model_path="models",
        max_workers=3,
        log_path=LOG_FILE
    )
    
    interface = WebInterface(config)
    interface.run(debug=True)
