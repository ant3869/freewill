# File: testing/test_model_responses.py
# Purpose: Test the fine-tuned model by evaluating responses to predefined prompts and logging outputs.

import os
import datetime
from llama_cpp import Llama

# Define paths
MODEL_PATH = "F:/lm-studio/models/QuantFactory/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-GGUF/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q8_0.gguf"
TEST_LOG_FILE = "./logs/test_model_responses.log"

# List of test prompts
test_prompts = [
    "Begin your thought process by reflecting on past reasoning.",
    "Explain why an autonomous system might consider speaking aloud.",
    "Reflect on mistakes and propose an opportunity to improve reasoning.",
    "Generate inner thoughts about exploring philosophical topics.",
    "Imagine a situation with no specific tasks. What thoughts arise?",
]

def log_test_result(prompt, response):
    """Log test results to a file."""
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    with open(TEST_LOG_FILE, "a") as log_file:
        log_file.write(f"{datetime.datetime.now()}\n")
        log_file.write(f"Prompt: {prompt}\n")
        log_file.write(f"Response: {response}\n")
        log_file.write("\n")

def test_model():
    """Run tests on the fine-tuned model and log results."""
    try:
        llama = Llama(model_path=MODEL_PATH)  # Initialize the model
        print("[INFO] Testing model with predefined prompts...")

        for idx, prompt in enumerate(test_prompts, 1):
            print(f"[INFO] Testing prompt {idx}/{len(test_prompts)}")
            try:
                response = llama(prompt, max_tokens=128, temperature=0.8, top_p=0.9)
                generated_text = response["choices"][0]["text"]

                print(f"Prompt {idx}: {prompt}")
                print(f"Response {idx}: {generated_text}\n")

                # Log the result
                log_test_result(prompt, generated_text)
            except Exception as e:
                error_message = f"Error generating response for prompt {idx}: {e}"
                print(error_message)
                log_test_result(prompt, error_message)

        print("[INFO] Testing completed. Check the log file for results.")

    except Exception as e:
        print(f"[ERROR] Model testing failed: {e}")
        with open(TEST_LOG_FILE, "a") as log_file:
            log_file.write(f"[ERROR] Model testing failed: {e}\n")

if __name__ == "__main__":
    test_model()
