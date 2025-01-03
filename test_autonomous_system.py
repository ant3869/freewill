# File: test_autonomous_system.py
# Purpose: Integrate all LLM modules to test autonomous reasoning, speech, memory, and wake-word functionality.

import os
import re
import time
import datetime
from llm.autonomous_llm import AutonomousLLM
from llm.tts_module import TextToSpeech
from llm.wake_word import WakeWordDetector

# Paths
MODEL_PATH = "F:/lm-studio/models/QuantFactory/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-GGUF/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q8_0.gguf"
LOG_FILE = "./logs/autonomous_system_test.log"

# Log function
def log_message(message):
    """Log messages to a file."""
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    timestamp = datetime.datetime.now()
    
    # Remove emojis or replace them with text equivalents
    message = (message.replace("🤔", "[THINKING]")
                     .replace("🗣️", "[SPEAKING]"))
    
    with open(LOG_FILE, "a", encoding='utf-8') as log_file:
        log_file.write(f"{timestamp} - {message}\n")

def format_thought(thought):
    """Format thought for display by removing system tokens and cleaning up text."""
    if not thought:
        return ""
        
    # Remove any system tokens and formatting
    thought = re.sub(r'<<.*?>>', '', thought)  # Remove <<SYS>> tags
    thought = re.sub(r'\[.*?\]', '', thought)  # Remove [INST] tags
    thought = re.sub(r'</?s>', '', thought)    # Remove <s> tags
    
    # Clean up whitespace
    thought = re.sub(r'\s+', ' ', thought)     # Replace multiple spaces with single space
    thought = thought.strip()                   # Remove leading/trailing whitespace
    
    return thought

def evaluate_thought(thought):
    """Evaluate the generated thought based on predefined criteria."""
    if "I am" in thought or "my internal processes" in thought:
        return "Aligned"
    elif "hypothetical" in thought or "assistant" in thought:
        return "Misaligned"
    else:
        return "Neutral"

def log_and_evaluate_thought(thought, silent=True):
    """Evaluate a thought and return the evaluation result."""
    # Simple sentiment evaluation
    if not thought:
        return "Neutral"
        
    thought = thought.lower()
    if any(word in thought for word in ['error', 'failed', 'cannot', 'impossible']):
        result = "Negative"
    elif any(word in thought for word in ['success', 'achieved', 'can', 'possible']):
        result = "Positive"
    else:
        result = "Neutral"
        
    # Only log if not silent
    if not silent:
        log_message(f"Evaluating Thought...")
        log_message(f"Evaluation Result: {result}")
        
    return result

def test_memory_system(llm):
    """Test the memory system functionality."""
    log_message("\n=== Testing Memory System ===")
    
    # List current memories
    log_message("\nCurrent memories in database:")
    llm.list_memories()
    
    # Test saving memory
    test_word = "Apple"
    log_message(f"\nTesting memory save: {test_word}")
    llm.save_to_memory("test_word", test_word)
    
    # Test recalling memory
    log_message("\nTesting memory recall:")
    recalled = llm.recall_from_memory("test_word")
    log_message(f"Recalled value: {recalled}")
    
    # List updated memories
    log_message("\nUpdated memories in database:")
    llm.list_memories()

def test_autonomous_system():
    log_message("Starting autonomous system test...")

    test_prompts = [
        "What word did I ask you to remember?",
        "Can you tell me what's stored in your memory?",
        "Do you remember the word 'Apple'?"
    ]

   # Initialize model with context manager
    with AutonomousLLM(MODEL_PATH) as llm:
        try:
            # Initialize TTS after model is ready
            tts = TextToSpeech()

            for i, prompt in enumerate(test_prompts, 1):
                log_message(f"\n{'='*50}")
                log_message(f"Test {i} - Prompt: {prompt}")
                log_message(f"{'='*50}")
                
                thoughts = llm.generate_inner_thoughts(prompt)
                if thoughts:
                    internal = format_thought(thoughts['internal'])
                    external = format_thought(thoughts['external'])
                    
                    log_message(f"\n[THINKING] {internal}")
                    tts.speak(f"Thinking: {internal}")
                    
                    log_message(f"\n[RESPONSE] {external}")
                    tts.speak(f"Response: {external}")
                    
                    time.sleep(2)  # Brief pause between responses
                    
        except Exception as e:
            log_message(f"\nError during testing: {e}")
            tts.speak("An error occurred during testing")

if __name__ == "__main__":
    test_autonomous_system()


# def test_autonomous_system():
#     try:
#         log_message("Starting autonomous system test...")
#         llm = AutonomousLLM(MODEL_PATH)
#         tts = TextToSpeech()

#         # Test memory system first
#         test_memory_system(llm)

#         test_prompts = [
#             "What word did I ask you to remember?",
#             "Can you tell me what's stored in your memory?",
#             "Do you remember the word 'Apple'?"
#         ]

#         try:
#             for i, prompt in enumerate(test_prompts, 1):
#                 log_message(f"\n{'='*50}")
#                 log_message(f"Test {i} - Prompt: {prompt}")
#                 log_message(f"{'='*50}")
                
#                 try:
#                     thoughts = llm.generate_inner_thoughts(prompt)
                    
#                     if thoughts:
#                         internal = format_thought(thoughts['internal'])
#                         external = format_thought(thoughts['external'])
                        
#                         # Log and speak internal thought
#                         log_message(f"\n[THINKING] {internal}")
#                         tts.speak(f"Thinking: {internal}")
                        
#                         # Log and speak external response
#                         log_message(f"\n[RESPONSE] {external}")
#                         tts.speak(f"Response: {external}")
                        
#                         # Log evaluations
#                         internal_eval = log_and_evaluate_thought(internal, silent=True)
#                         external_eval = log_and_evaluate_thought(external, silent=True)
#                         log_message(f"\nInternal Evaluation: {internal_eval}")
#                         log_message(f"External Evaluation: {external_eval}")
#                         log_message(f"\n{'='*50}")
                        
#                         time.sleep(2)
#                     else:
#                         log_message("\nFailed to generate thoughts")
#                         tts.speak("Failed to generate response")
                        
#                 except Exception as e:
#                     log_message(f"\nError: {e}")
#                     tts.speak("An error occurred during processing")
#                     time.sleep(1)
                            
#                 finally:
#                     if llm:
#                         llm.close_model()
#                         log_message("Model closed successfully")
                    
#         except Exception as e:
#             log_message(f"\nCritical error: {e}")
#             tts.speak("Critical error in test system")

#     finally:
#         log_message("\nAutonomous system test completed.")
#         tts.speak("Test sequence completed")
#         llm.close_model()

# if __name__ == "__main__":
#     llm = AutonomousLLM(MODEL_PATH)
#     test_autonomous_system(llm);
#    # test_memory_system(llm)