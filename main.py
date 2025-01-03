# File: main.py
# Purpose: Main entry point for the autonomous LLM application.

import os
from llm.autonomous_llm import AutonomousLLM
from llm.tts_module import TTSModule
from interface import WebInterface, ServerConfig

def get_default_model_path():
    """Get the default model path from environment or fallback to common locations"""
    # Check environment variable first
    model_path = os.getenv('FREEWILL_MODEL_PATH')
    if model_path and os.path.exists(model_path):
        return model_path
        
    # Common locations to check
    common_paths = [
        'models/DarkIdol-Llama-3_1.gguf',
        os.path.expanduser('~/models/DarkIdol-Llama-3_1.gguf'),
        'models/DarkIdol-Llama-3_1.gguf'
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
            
    return None

def main():
    # Get model path
    model_path = get_default_model_path()
    if not model_path:
        model_path = 'models/DarkIdol-Llama-3_1.gguf'  # Default to local path even if it doesn't exist yet

    # Configure the server
    config = ServerConfig(
        model_path='models/DarkIdol-Llama-3_1.gguf',
        models_directory="models",
        max_workers=3,
        log_path='logs'
    )
    
    # Create and run the interface
    interface = WebInterface(config)
    interface.run(debug=True)

if __name__ == "__main__":
    main()