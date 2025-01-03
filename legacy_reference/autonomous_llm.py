from llama_cpp import Llama
import torch
from llm.memory_database import MemoryDatabase
import datetime
import atexit
import os
import re
import logging

# Set up logger
logger = logging.getLogger(__name__)

MODEL_PATH = "./models/DarkIdol-Llama-3_1.gguf"
LOG_FILE = "./logs/autonomous_system_test.log"
DEFAULT_MAIN_PROMPT = "You are a helpful AI assistant."
DEFAULT_INTERNAL_PROMPT = "[INTERNAL PROCESSING]"
DEFAULT_EXTERNAL_PROMPT = "[EXTERNAL RESPONSE]"

# def log_message(message):
#     """Log messages to a file."""
#     if not os.path.exists("./logs"):
#         os.makedirs("./logs")
#     timestamp = datetime.datetime.now()
    
#     # Remove emojis or replace them with text equivalents
#     message = (message.replace("🤔", "[THINKING]")
#                      .replace("🗣️", "[SPEAKING]"))
    
#     with open(LOG_FILE, "a", encoding='utf-8') as log_file:
#         log_file.write(f"{timestamp} - {message}\n")

class AutonomousLLM:
    def __init__(self, model_path, max_len=512, memory_db_path="memory.db"):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing AutonomousLLM...")

        self.model_path = model_path
        self.max_len = max_len
        self.memory = MemoryDatabase(memory_db_path)
        self._llama = None 

        # Register cleanup on program exit
        atexit.register(self.cleanup)
        
        # Check for CUDA (GPU) availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")

        if self.device == "cuda":
            gpu_info = torch.cuda.get_device_properties(0)
            self.logger.info(f"GPU: {gpu_info.name}")
            self.logger.info(f"Memory: {gpu_info.total_memory / 1024**3:.2f} GB")

        try:
            # Initialize model with GPU settings
            self._llama = Llama(
                model_path=model_path,
                n_gpu_layers=-1,  # -1 means use all layers on GPU
                n_ctx=4096,       # Context window
                n_batch=512,      # Batch size for prompt processing
                use_mmap=True,    # Memory mapping for faster loading
                use_mlock=False,  # Don't lock memory
                verbose=True      # Show loading progress
            )
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
        
  
        
        # # Initialize the model
        # self.initialize_model()
    
    # def initialize_model(self):
    #     """Initialize the LLaMA model."""
    #     try:
    #         if self._llama is None:
    #             self._llama = Llama(
    #                 model_path=self.model_path,
    #                 n_ctx=self.max_len,
    #                 n_threads=8,
    #                 verbose=False  # Reduce output noise
    #             )
    #     except Exception as e:
    #         log_message(f"Error initializing model: {e}")
    #         self._llama = None

    def clean_response(self, text):
        """Clean up response text by removing system tokens and extra whitespace."""
        if not text:
            return ""
            
        # Remove system tokens and formatting
        patterns = [
            r'\[INST\].*?\[/INST\]',
            r'<<SYS>>.*?<</SYS>>',
            r'<s>|</s>',
            r'\[/\w+\]'
        ]
        
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)
        
        # Fix common text issues
        replacements = [
            (r'\bhe\b(?!\w)', 'the'),  # Fix 'he' -> 'the' when it's a standalone word
            (r'\bi\b(?!\w)', 'I'),     # Capitalize standalone 'i'
            (r'\b\'m\b', "'m"),        # Fix 'm -> 'm
            (r'\bn\b', 'in'),          # Fix 'n' -> 'in'
            (r'(?<=\w)\'(?=\w)', "'")  # Fix apostrophes in contractions
        ]
        
        for old, new in replacements:
            cleaned = re.sub(old, new, cleaned)
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())      
        return cleaned.strip()

    def generate_inner_thoughts(self, prompt):
        """Generate both internal thoughts and external response."""
        if not self._llama:
            self.initialize_model()

        try:
            self.logger.info(f"Starting generation for prompt: {prompt}")

            system_prompt = """You are a helpful AI assistant. For each input, you will:
                1. Process input internally (private) (shown as [Internal thought])
                2. Provide an external response (public) (shown as [External response])
                Please keep responses concise and natural."""

            # Format the complete prompt
            formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\n\n[Internal thought]:"

            self.logger.info("Generating internal thought...")
            internal_response = self._llama(
                formatted_prompt,
                max_tokens=256,
                temperature=0.7,
                top_p=0.9,
                stop=["[External response]", "User:"],  # Stop tokens
                echo=False
            )

            if not internal_response or not internal_response['choices']:
                raise Exception("No internal response generated")
                
            internal_thought = internal_response['choices'][0]['text'].strip()
            self.logger.info(f"Internal thought generated: {internal_thought[:100]}...")

            # Generate external response
            external_prompt = f"{formatted_prompt}\n{internal_thought}\n\n[External response]:"
            
            self.logger.info("Generating external response...")
            external_response = self._llama(
                external_prompt,
                max_tokens=256,
                temperature=0.7,
                top_p=0.9,
                stop=["User:", "[Internal thought]"],  # Stop tokens
                echo=False
            )

            if not external_response or not external_response['choices']:
                raise Exception("No external response generated")
            
            external_thought = external_response['choices'][0]['text'].strip()
            self.logger.info(f"External response generated: {external_thought[:100]}...")

            return {
                'internal': internal_thought,
                'external': external_thought
            }
        
        except Exception as e:
            self.logger.error(f"Error in generate_inner_thoughts: {str(e)}", exc_info=True)
            self.logger.error(f"Prompt that caused error: {prompt}")
            self.logger.error("Model state:", self._llama is not None)
            
            # Try to get GPU memory info if available
            if torch.cuda.is_available():
                try:
                    mem_info = {
                        'allocated': f"{torch.cuda.memory_allocated()/1024**3:.2f}GB",
                        'reserved': f"{torch.cuda.memory_reserved()/1024**3:.2f}GB",
                        'max_allocated': f"{torch.cuda.max_memory_allocated()/1024**3:.2f}GB"
                    }
                    self.logger.error(f"GPU Memory state: {mem_info}")
                except:
                    pass
            
            return None
               
    def __call__(self, prompt, **kwargs):
        """Wrapper for the model's create_completion method with better error handling"""
        try:
            if not self._llama:
                raise Exception("Model not initialized")
                
            response = self._llama.create_completion(
                prompt=prompt,
                **kwargs
            )
            return response
        except Exception as e:
            self.logger.error(f"Error in model call: {str(e)}")
            return None

    def save_to_memory(self, key, value):
        """Save a key-value pair to memory."""
        try:
            self.memory.save_memory(key, value)
            self.logger.info(f"Saved to memory: {key} = {value}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving to memory: {e}")
            return False
            
    def recall_from_memory(self, key):
        """Recall a value from memory by key."""
        try:
            value = self.memory.get_memory(key)
            self.logger.info(f"Recalled from memory: {key} = {value}")
            return value
        except Exception as e:
            self.logger.error(f"Error recalling from memory: {e}")
            return None
            
    def list_memories(self):
        """List all stored memories."""
        try:
            memories = self.memory.get_all_memories()
            self.logger.info("\nStored Memories:")
            for key, value, timestamp in memories:
                self.logger.info(f"- {key}: {value} (stored at {timestamp})")
            return memories
        except Exception as e:
            self.logger.error(f"Error listing memories: {e}")
            return []

    def recall_recent_thoughts(self, limit=5):
        """Retrieve recent thoughts from the memory database."""
        return self.memory.retrieve_recent_thoughts(limit)

    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, '_llama') and self._llama is not None:
                # Remove the reference and let Python handle cleanup
                self._llama = None
        except Exception as e:
            pass  # Suppress cleanup errors

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def __del__(self):
        """Destructor."""
        self.cleanup()

    def stop_generation(self):
        """Stop the current generation process."""
        try:
            if self._llama:
                # Implement model-specific cancellation
                # This will depend on your LLM implementation
                self._llama.stop_generation()  # or similar method
                return True
        except Exception as e:
            logger.error(f"Error stopping generation: {e}")
        return False
    
    def close_model(self):
        try:
            if self._llama:
                # Clean up GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._llama = None
                self.logger.info("Model closed and resources freed")
        except Exception as e:
            self.logger.error(f"Error closing model: {str(e)}")

    def update_prompts(self, main=None, internal=None, external=None):
        """Update system prompts, using defaults if None provided"""
        logger.info("Updating system prompts")
        self.main_prompt = main if main else self.main_prompt
        self.internal_prompt = internal if internal else self.internal_prompt
        self.external_prompt = external if external else self.external_prompt
        logger.info(f"Prompts updated - Main: {len(self.main_prompt)} chars, Internal: {len(self.internal_prompt)} chars, External: {len(self.external_prompt)} chars")

    def generate_response(self, prompt, use_internal=True):
        if not self.model:
            raise Exception("Model not loaded")
            
        try:
            full_prompt = f"{self.main_prompt}\n\n"
            
            if use_internal:
                full_prompt += f"{self.internal_prompt}\n"
                internal_result = self.model.create_completion(
                    full_prompt + prompt,
                    max_tokens=512,
                    temperature=0.7,
                    stop=["</output>", "\n\n"]
                )
                internal_response = internal_result['choices'][0]['text']
                
                full_prompt += f"{internal_response}\n{self.external_prompt}\n"
            
            result = self.model.create_completion(
                full_prompt + prompt,
                max_tokens=512,
                temperature=0.7,
                stop=["</output>", "\n\n"]
            )
            
            return result['choices'][0]['text']
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
            
    def load_model(self, model_path):
        """Load the model from path"""
        try:
            from llama_cpp import Llama
            
            # Initialize the model with your parameters
            model = Llama(
                model_path=model_path,
                n_ctx=2048,  # Adjust context window as needed
                n_threads=4   # Adjust based on your CPU
            )
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise