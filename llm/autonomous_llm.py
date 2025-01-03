from llama_cpp import Llama
import torch
import datetime
import atexit
import re
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from llm.memory_database import MemoryDatabase

@dataclass
class ModelConfig:
    n_ctx: int = 4096
    n_batch: int = 512
    n_gpu_layers: int = -1
    use_mmap: bool = True
    use_mlock: bool = False
    verbose: bool = True

DEFAULT_MAIN_PROMPT = "You are a helpful AI assistant."
DEFAULT_INTERNAL_PROMPT = "[INTERNAL PROCESSING]"
DEFAULT_EXTERNAL_PROMPT = "[EXTERNAL RESPONSE]"

class AutonomousLLM:
    def __init__(
        self, 
        model_path: str,
        config: Optional[ModelConfig] = None,
        memory_db_path: str = "memory.db"
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing AutonomousLLM...")
        
        self.model_path = model_path
        self.config = config or ModelConfig()
        self._llama = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.memory = MemoryDatabase(memory_db_path)
        
        self.main_prompt = DEFAULT_MAIN_PROMPT
        self.internal_prompt = DEFAULT_INTERNAL_PROMPT
        self.external_prompt = DEFAULT_EXTERNAL_PROMPT
        
        # Initialize model
        self._initialize_model()
        
        # Register cleanup
        atexit.register(self.cleanup)

    def _initialize_model(self) -> None:
        """Initialize the LLaMA model with proper error handling."""
        try:
            if self.device == "cuda":
                gpu_info = torch.cuda.get_device_properties(0)
                self.logger.info(f"Using GPU: {gpu_info.name} with {gpu_info.total_memory / 1024**3:.2f} GB")

            self._llama = Llama(
                model_path=self.model_path,
                n_ctx=self.config.n_ctx,
                n_batch=self.config.n_batch,
                n_gpu_layers=self.config.n_gpu_layers,
                use_mmap=self.config.use_mmap,
                use_mlock=self.config.use_mlock,
                verbose=self.config.verbose
            )
            self.logger.info("Model initialized successfully")
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def clean_response(self, text: str) -> str:
        """Clean up response text by removing system tokens and extra whitespace."""
        if not text:
            return ""
            
        patterns = [
            r'\[INST\].*?\[/INST\]',
            r'<<SYS>>.*?<</SYS>>',
            r'<s>|</s>',
            r'\[/\w+\]'
        ]
        
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)
        
        replacements = [
            (r'\bhe\b(?!\w)', 'the'),
            (r'\bi\b(?!\w)', 'I'),
            (r'\b\'m\b', "'m"),
            (r'\bn\b', 'in'),
            (r'(?<=\w)\'(?=\w)', "'")
        ]
        
        for old, new in replacements:
            cleaned = re.sub(old, new, cleaned)
            
        return ' '.join(cleaned.split()).strip()

    def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response using the model with proper error handling."""
        if not self._llama:
            self._initialize_model()

        try:
            response = self._llama(
                prompt,
                max_tokens=kwargs.get('max_tokens', 512),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                stop=kwargs.get('stop', ["</output>", "\n\n"]),
                echo=kwargs.get('echo', False)
            )
            
            if not response or not response.get('choices'):
                raise ValueError("No response generated")
                
            return {
                'response': self.clean_response(response['choices'][0]['text']),
                'raw': response
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            if self.device == "cuda":
                self._log_gpu_state()
            raise

    def _log_gpu_state(self) -> None:
        """Log GPU memory state for debugging."""
        if torch.cuda.is_available():
            try:
                self.logger.info({
                    'allocated': f"{torch.cuda.memory_allocated()/1024**3:.2f}GB",
                    'reserved': f"{torch.cuda.memory_reserved()/1024**3:.2f}GB",
                    'max_allocated': f"{torch.cuda.max_memory_allocated()/1024**3:.2f}GB"
                })
            except Exception as e:
                self.logger.error(f"Failed to log GPU state: {str(e)}")

    def cleanup(self) -> None:
        """Clean up resources properly."""
        try:
            if self._llama:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._llama = None
                self.logger.info("Model cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def __del__(self):
        """Destructor."""
        self.cleanup()

    def save_to_memory(self, key: str, value: str) -> bool:
        try:
            self.memory.save_memory(key, value)
            self.logger.info(f"Saved to memory: {key} = {value}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving to memory: {e}")
            return False
            
    def recall_from_memory(self, key: str) -> Optional[str]:
        try:
            value = self.memory.get_memory(key)
            self.logger.info(f"Recalled from memory: {key} = {value}")
            return value
        except Exception as e:
            self.logger.error(f"Error recalling from memory: {e}")
            return None
            
    def list_memories(self):
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
        return self.memory.retrieve_recent_thoughts(limit)

    def update_prompts(self, main: Optional[str] = None, internal: Optional[str] = None, external: Optional[str] = None) -> None:
        self.logger.info("Updating system prompts")
        if main is not None:
            self.main_prompt = main
        if internal is not None:
            self.internal_prompt = internal
        if external is not None:
            self.external_prompt = external
        self.logger.info(f"Prompts updated - Main: {len(self.main_prompt)} chars, Internal: {len(self.internal_prompt)} chars, External: {len(self.external_prompt)} chars")

    def stop_generation(self) -> bool:
        try:
            if self._llama:
                # Implement model-specific cancellation
                self._llama.stop_generation()
                return True
        except Exception as e:
            self.logger.error(f"Error stopping generation: {e}")
        return False
