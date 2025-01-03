import os
import datetime
from datasets import load_dataset, Dataset
from llama_cpp import Llama
from datasets import load_from_disk

# Paths
MODEL_PATH = os.path.abspath("F:/lm-studio/models/QuantFactory/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-GGUF/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q8_0.gguf")
PREPARED_DATASET_PATH = os.path.abspath("./training/dataset/prepared_dataset.jsonl")
TOKENIZED_DATASET_PATH = os.path.abspath("./training/dataset/tokenized_dataset")
PROCESSED_TEXT_DIR = os.path.abspath("./training/dataset/processed/")
LOG_FILE = os.path.abspath("./logs/prepare_dataset.log")

# Logging utility
def log_message(message):
    os.makedirs("./logs", exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(f"{datetime.datetime.now()} - {message}\n")

# Fine-tuning function
def fine_tune(dataset_path, model_path, max_length=512, test_size=0.1):
    log_message("Initializing Llama model for fine-tuning...")
    llama = Llama(model_path=model_path)
    try:
        # Load the tokenized dataset
        log_message(f"Loading tokenized dataset from '{dataset_path}'...")
        dataset = load_from_disk(dataset_path)

        # Split into training and testing sets
        log_message("Splitting dataset into training and testing sets...")
        split_dataset = dataset.train_test_split(test_size=test_size)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]

        # Fine-tuning loop
        log_message("Starting fine-tuning...")
        for idx, example in enumerate(train_dataset):

            def clean_text(text):
                return text.encode("utf-8", errors="ignore").decode("utf-8")

            # Use the plain text input from the dataset
            prompt = clean_text(example["text"][:max_length])

            # Truncate if necessary to fit within the context window
            if len(prompt) > max_length:
                prompt = prompt[:max_length - 10]  # Reserve 10 tokens for generation

            try:
                response = llama(
                    prompt=prompt,
                    max_tokens=10,  # Adjust as needed
                    temperature=0.8,
                    top_p=0.9
                )
                generated_text = response.get("choices", [{}])[0].get("text", "")
                log_message(f"Generated Response: {generated_text}")
            except UnicodeEncodeError as e:
                log_message(f"Encoding error: {e}. Skipping this example.")

        # Evaluation loop
        log_message("Starting evaluation...")
        for idx, example in enumerate(test_dataset):
            # Use the plain text input from the dataset
            prompt = example["text"]

            # Truncate if necessary to fit within the context window
            if len(prompt) > max_length:
                prompt = prompt[:max_length - 10]

            response = llama(
                prompt=prompt,
                max_tokens=10,  # Adjust as needed
                temperature=0.8,
                top_p=0.9
            )
            generated_text = response.get("choices", [{}])[0].get("text", "")
            log_message(f"Test Example {idx}: Prompt: {prompt} | Generated Response: {generated_text}")

        log_message("Fine-tuning and evaluation completed successfully.")
    except Exception as e:
        log_message(f"Error during fine-tuning or evaluation: {e}")
        raise
    finally:
        llama.close()
        log_message("Llama model resources released.")


# Main entry point
def main():
    try:
        log_message("Fine-tuning process started.")
        fine_tune(TOKENIZED_DATASET_PATH, MODEL_PATH)
        log_message("Fine-tuning process completed successfully.")
    except Exception as e:
        log_message(f"Critical error: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

# # File: training/fine_tune.py
# # Purpose: Fine-tunes and evaluates the LLM model on a prepared dataset.

# import os
# import json
# import datetime
# from llama_cpp import Llama

# # Define paths
# MODEL_PATH = "F:/lm-studio/models/QuantFactory/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-GGUF/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q8_0.gguf"
# PREPARED_DATASET_PATH = "./training/dataset/prepared_dataset.jsonl"
# LOG_FILE = "./logs/fine_tune.log"

# # Adjustable parameters
# CONTEXT_WINDOW = 2048  # Desired context window size (adjust as per your model's capability)
# MAX_GENERATION_TOKENS = 64  # Tokens to generate during evaluation

# # Logging utility
# def log_message(message):
#     if not os.path.exists("./logs"):
#         os.makedirs("./logs")
#     with open(LOG_FILE, "a") as log_file:
#         log_file.write(f"{datetime.datetime.now()} - {message}\n")

# # Validate tokenized dataset path
# if not os.path.exists(TOKENIZED_DATASET_PATH):
#     raise FileNotFoundError(f"Tokenized dataset path '{TOKENIZED_DATASET_PATH}' does not exist.")

# def truncate_prompt(prompt, max_tokens):
#     """
#     Truncate the prompt to fit within the allowed context window.
#     """
#     if len(prompt) > max_tokens:
#         prompt = prompt[:max_tokens]
#     return prompt

# def main(test_size=0.1):
#     # Load the tokenized dataset
#     print(f"[INFO] Loading tokenized dataset from '{TOKENIZED_DATASET_PATH}'...")
#     try:
#         tokenized_datasets = load_from_disk(TOKENIZED_DATASET_PATH)
#     except Exception as e:
#         raise RuntimeError(f"Error loading tokenized dataset: {e}")

#     # Check dataset size for splitting
#     min_samples = max(2, int(1 / test_size))
#     if len(tokenized_datasets) < min_samples:
#         raise ValueError(
#             f"Not enough samples in the dataset to perform a train-test split. "
#             f"Minimum required samples: {min_samples}, but got {len(tokenized_datasets)}."
#         )

#     # Split dataset into train-test sets
#     print("[INFO] Splitting dataset into training and testing sets...")
#     split_dataset = tokenized_datasets.train_test_split(test_size=test_size)
#     train_dataset = split_dataset["train"]
#     test_dataset = split_dataset["test"]

#     # Initialize Llama model
#     print(f"[INFO] Initializing the Llama model with a context window of {CONTEXT_WINDOW} tokens...")
#     llama = Llama(model_path=MODEL_PATH, n_ctx=CONTEXT_WINDOW)

#     # Fine-tuning loop
#     print("[INFO] Starting fine-tuning...")
#     for idx, example in enumerate(train_dataset):
#         prompt = truncate_prompt(example["text"], CONTEXT_WINDOW - MAX_GENERATION_TOKENS)
#         response = llama(prompt, max_tokens=MAX_GENERATION_TOKENS, temperature=0.8, top_p=0.9)
#         generated_text = response["choices"][0]["text"]
#         log_message(f"Train Prompt: {prompt}\nGenerated: {generated_text}\n")

#     # Evaluation loop
#     print("[INFO] Starting evaluation...")
#     try:
#         for idx, example in enumerate(test_dataset):
#             prompt = truncate_prompt(example["text"], CONTEXT_WINDOW - MAX_GENERATION_TOKENS)
#             response = llama(prompt, max_tokens=MAX_GENERATION_TOKENS, temperature=0.8, top_p=0.9)
#             generated_text = response["choices"][0]["text"]

#             log_message(f"Test Prompt: {prompt}\nGenerated: {generated_text}\n")

#         print("[INFO] Evaluation complete.")
#     except Exception as e:
#         log_message(f"Evaluation failed: {e}")
#         raise RuntimeError(f"Evaluation failed: {e}")

#     finally:
#         # Explicitly close the Llama instance
#         llama.close()
#         print("[INFO] Model resources released.")


# if __name__ == "__main__":
#     try:
#         log_message("Fine-tuning started.")
#         main()
#         log_message("Fine-tuning completed successfully.")
#     except Exception as e:
#         log_message(f"Critical error: {e}")
#         print(f"An error occurred: {e}")

# # File: training/fine_tune.py
# # Purpose: Fine-tunes and evaluates the LLM model on a tokenized dataset.

# import os
# import datetime
# from datasets import load_from_disk, DatasetDict
# from llama_cpp import Llama

# # Define paths
# MODEL_PATH = "F:/lm-studio/models/QuantFactory/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-GGUF/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q8_0.gguf"
# TOKENIZED_DATASET_PATH = "./training/dataset/tokenized_dataset"
# LOG_FILE = "./logs/evaluate.log"

# # Logging utility
# def log_message(message):
#     if not os.path.exists("./logs"):
#         os.makedirs("./logs")
#     with open(LOG_FILE, "a") as log_file:
#         log_file.write(f"{datetime.datetime.now()} - {message}\n")

# # Validate tokenized dataset path
# if not os.path.exists(TOKENIZED_DATASET_PATH):
#     raise FileNotFoundError(f"Tokenized dataset path '{TOKENIZED_DATASET_PATH}' does not exist.")

# def main(test_size=0.1):
#     # Load the tokenized dataset
#     print(f"[INFO] Loading tokenized dataset from '{TOKENIZED_DATASET_PATH}'...")
#     try:
#         tokenized_datasets = load_from_disk(TOKENIZED_DATASET_PATH)
#     except Exception as e:
#         raise RuntimeError(f"Error loading tokenized dataset: {e}")

#     # Check dataset size for splitting
#     min_samples = max(2, int(1 / test_size))
#     if len(tokenized_datasets) < min_samples:
#         raise ValueError(
#             f"Not enough samples in the dataset to perform a train-test split. "
#             f"Minimum required samples: {min_samples}, but got {len(tokenized_datasets)}."
#         )

#     # Split dataset into train-test sets
#     print("[INFO] Splitting dataset into training and testing sets...")
#     split_dataset = tokenized_datasets.train_test_split(test_size=test_size)
#     train_dataset = split_dataset["train"]
#     test_dataset = split_dataset["test"]

#     # Initialize Llama model
#     print("[INFO] Initializing the Llama model...")
#     llama = Llama(model_path=MODEL_PATH)

#     # Fine-tuning loop
#     print("[INFO] Starting fine-tuning...")
#     for idx, example in enumerate(train_dataset):
#         prompt = example["text"]
#         response = llama(prompt, max_tokens=64, temperature=0.8, top_p=0.9)
#         generated_text = response["choices"][0]["text"]
#         log_message(f"Train Prompt: {prompt}\nGenerated: {generated_text}\n")

#     # Evaluation loop
#     print("[INFO] Starting evaluation...")
#     try:
#         for idx, example in enumerate(test_dataset):
#             prompt = example["text"]
#             response = llama(prompt, max_tokens=64, temperature=0.8, top_p=0.9)
#             generated_text = response["choices"][0]["text"]

#             log_message(f"Test Prompt: {prompt}\nGenerated: {generated_text}\n")

#         print("[INFO] Evaluation complete.")
#     except Exception as e:
#         log_message(f"Evaluation failed: {e}")
#         raise RuntimeError(f"Evaluation failed: {e}")

#     finally:
#         # Explicitly close the Llama instance
#         llama.close()
#         print("[INFO] Model resources released.")


# if __name__ == "__main__":
#     try:
#         log_message("Evaluation started.")
#         main()
#         log_message("Evaluation completed successfully.")
#     except Exception as e:
#         log_message(f"Critical error: {e}")
#         print(f"An error occurred: {e}")
