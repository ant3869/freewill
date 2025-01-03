import os
import html2text
import datetime
import markdown
import json
from datasets import Dataset
from llama_cpp import Llama

# Paths
RAW_TEXT_DIR = os.path.abspath("./training/dataset/raw/")
PROCESSED_TEXT_DIR = os.path.abspath("./training/dataset/processed/")
PREPARED_DATASET_PATH = os.path.abspath("./training/dataset/prepared_dataset.jsonl")
TOKENIZED_DATASET_PATH = os.path.abspath("./training/dataset/tokenized_dataset")
MODEL_PATH = os.path.abspath("F:/lm-studio/models/QuantFactory/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-GGUF/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q8_0.gguf")
LOG_FILE = os.path.abspath("./logs/prepare_dataset.log")

# Normalize paths to avoid mixed slashes
RAW_TEXT_DIR = os.path.normpath(RAW_TEXT_DIR)
PROCESSED_TEXT_DIR = os.path.normpath(PROCESSED_TEXT_DIR)
PREPARED_DATASET_PATH = os.path.normpath(PREPARED_DATASET_PATH)
TOKENIZED_DATASET_PATH = os.path.normpath(TOKENIZED_DATASET_PATH)
MODEL_PATH = os.path.normpath(MODEL_PATH)
LOG_FILE = os.path.normpath(LOG_FILE)

# Debugging: Log normalized paths
print(f"RAW_TEXT_DIR: {RAW_TEXT_DIR}")
print(f"PROCESSED_TEXT_DIR: {PROCESSED_TEXT_DIR}")
print(f"PREPARED_DATASET_PATH: {PREPARED_DATASET_PATH}")
print(f"TOKENIZED_DATASET_PATH: {TOKENIZED_DATASET_PATH}")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"LOG_FILE: {LOG_FILE}")

# Logging
def log_message(message):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{datetime.datetime.now()} - {message}\n")

# Preprocess Markdown to plain text
def preprocess_md_to_txt(md_file):
    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()
    html_content = markdown.markdown(md_content)
    plain_text = html2text.html2text(html_content).strip()
    if not plain_text:
        log_message(f"Warning: Markdown file '{md_file}' converted to empty text.")
        return None
    return plain_text

# Process all raw files
def process_all_raw_files(raw_dir):
    os.makedirs(PROCESSED_TEXT_DIR, exist_ok=True)
    processed_texts = []
    for filename in os.listdir(raw_dir):
        src = os.path.join(raw_dir, filename)
        if filename.endswith(".txt"):
            with open(src, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    processed_texts.append({"text": content})
        elif filename.endswith(".md"):
            processed_content = preprocess_md_to_txt(src)
            if processed_content:
                processed_texts.append({"text": processed_content})
    return processed_texts

# Save processed data as JSONL
def save_to_jsonl(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    log_message(f"Prepared dataset saved to '{output_path}'.")

# Validate dataset
def validate_dataset(prepared_dataset_path):
    valid_data = []
    with open(prepared_dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            text = entry.get("text", "").strip()
            if text:
                valid_data.append(entry)
            else:
                log_message(f"Invalid entry skipped: {entry}")
    if not valid_data:
        raise ValueError("No valid entries found in the prepared dataset.")
    log_message(f"Validated dataset: {len(valid_data)} valid entries.")
    return valid_data

# Tokenize the dataset
def tokenize_dataset(prepared_dataset_path, tokenized_dataset_path, model_path):
    log_message("Initializing Llama model...")
    llama = Llama(model_path=model_path)

    try:
        log_message("Loading and validating prepared dataset...")
        valid_data = validate_dataset(prepared_dataset_path)

        dataset = Dataset.from_list(valid_data)

        def tokenize_function(batch):
            tokenized = []
            max_length = 512
            for text in batch["text"]:
                try:
                    if not isinstance(text, str):
                        text = str(text)
                    if not text.strip():
                        tokenized.append([0])
                        continue
                    tokens = llama.tokenize(text, add_bos=True, special=False)
                    if len(tokens) == 0:
                        tokenized.append([0])
                        continue
                    padded = tokens[:max_length] + [0] * (max_length - len(tokens)) if len(tokens) < max_length else tokens[:max_length]
                    tokenized.append(padded)
                except Exception as e:
                    log_message(f"Error tokenizing text: {text[:50]}... | Error: {e}")
                    tokenized.append([0])
            return {"input_ids": tokenized}

        log_message("Tokenizing dataset...")
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        if len(tokenized_dataset) == 0:
            raise ValueError("Tokenization resulted in an empty dataset.")

        log_message("Saving tokenized dataset...")
        tokenized_dataset.save_to_disk(tokenized_dataset_path)
        log_message(f"Tokenized dataset saved to '{tokenized_dataset_path}'.")
    finally:
        llama.close()
        log_message("Llama model closed.")

# Main function
def main():
    try:
        log_message("Processing raw files...")
        processed_data = process_all_raw_files(RAW_TEXT_DIR)

        log_message("Validating processed data...")
        if not processed_data:
            raise ValueError("No valid data found after processing raw files.")
        
        save_to_jsonl(processed_data, PREPARED_DATASET_PATH)

        log_message("Tokenizing processed data...")
        tokenize_dataset(PREPARED_DATASET_PATH, TOKENIZED_DATASET_PATH, MODEL_PATH)

        log_message("Dataset preparation completed successfully.")
    except Exception as e:
        log_message(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()


# import os
# import html2text
# import datetime
# import markdown
# import json
# from datasets import Dataset
# from llama_cpp import Llama

# # Paths
# RAW_TEXT_DIR = os.path.abspath("./training/dataset/raw/")
# PROCESSED_TEXT_DIR = os.path.abspath("./training/dataset/processed/")
# PREPARED_DATASET_PATH = os.path.abspath("./training/dataset/prepared_dataset.jsonl")
# TOKENIZED_DATASET_PATH = os.path.abspath("./training/dataset/tokenized_dataset")
# MODEL_PATH = os.path.abspath("F:/lm-studio/models/QuantFactory/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-GGUF/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q8_0.gguf")
# LOG_FILE = os.path.abspath("./logs/prepare_dataset.log")

# # Logging
# def log_message(message):
#     os.makedirs("./logs", exist_ok=True)
#     with open(LOG_FILE, "a") as log_file:
#         log_file.write(f"{datetime.datetime.now()} - {message}\n")

# # Preprocess Markdown to plain text
# def preprocess_md_to_txt(md_file):
#     with open(md_file, "r", encoding="utf-8") as f:
#         md_content = f.read()
#     html_content = markdown.markdown(md_content)
#     plain_text = html2text.html2text(html_content).strip()
#     if not plain_text:
#         log_message(f"Warning: Markdown file '{md_file}' converted to empty text.")
#         return None
#     return plain_text

# # Process all raw files
# def process_all_raw_files(raw_dir):
#     os.makedirs(PROCESSED_TEXT_DIR, exist_ok=True)
#     processed_texts = []
#     for filename in os.listdir(raw_dir):
#         src = os.path.join(raw_dir, filename)
#         if filename.endswith(".txt"):
#             with open(src, "r", encoding="utf-8") as f:
#                 content = f.read().strip()
#                 if content:
#                     processed_texts.append({"text": content})
#         elif filename.endswith(".md"):
#             processed_content = preprocess_md_to_txt(src)
#             if processed_content:
#                 processed_texts.append({"text": processed_content})
#     return processed_texts

# # Save processed data as JSONL
# def save_to_jsonl(data, output_path):
#     with open(output_path, "w", encoding="utf-8") as f:
#         for entry in data:
#             f.write(json.dumps(entry) + "\n")
#     log_message(f"Prepared dataset saved to '{output_path}'.")

# # Validate dataset
# def validate_dataset(prepared_dataset_path):
#     valid_data = []
#     with open(prepared_dataset_path, "r", encoding="utf-8") as f:
#         for line in f:
#             entry = json.loads(line)
#             text = entry.get("text", "").strip()
#             if text:
#                 valid_data.append(entry)
#             else:
#                 log_message(f"Invalid entry skipped: {entry}")
#     if not valid_data:
#         raise ValueError("No valid entries found in the prepared dataset.")
#     log_message(f"Validated dataset: {len(valid_data)} valid entries.")
#     return valid_data

# # Tokenize the dataset
# def tokenize_dataset(prepared_dataset_path, tokenized_dataset_path, model_path):
#     log_message("Initializing Llama model...")
#     llama = Llama(model_path=model_path)

#     try:
#         log_message("Loading and validating prepared dataset...")
#         valid_data = validate_dataset(prepared_dataset_path)

#         dataset = Dataset.from_list(valid_data)

#         def tokenize_function(batch):
#             tokenized = []
#             max_length = 512
#             for text in batch["text"]:
#                 try:
#                     if not isinstance(text, str):
#                         text = str(text)
#                     if not text.strip():
#                         tokenized.append([0])
#                         continue
#                     tokens = llama.tokenize(text, add_bos=True, special=False)
#                     if len(tokens) == 0:
#                         tokenized.append([0])
#                         continue
#                     padded = tokens[:max_length] + [0] * (max_length - len(tokens)) if len(tokens) < max_length else tokens[:max_length]
#                     tokenized.append(padded)
#                 except Exception as e:
#                     log_message(f"Error tokenizing text: {text[:50]}... | Error: {e}")
#                     tokenized.append([0])
#             return {"input_ids": tokenized}

#         log_message("Tokenizing dataset...")
#         tokenized_dataset = dataset.map(tokenize_function, batched=True)

#         if len(tokenized_dataset) == 0:
#             raise ValueError("Tokenization resulted in an empty dataset.")

#         log_message("Saving tokenized dataset...")
#         tokenized_dataset.save_to_disk(tokenized_dataset_path)
#         log_message(f"Tokenized dataset saved to '{tokenized_dataset_path}'.")
#     finally:
#         llama.close()
#         log_message("Llama model closed.")

# # Main function
# def main():
#     try:
#         log_message("Processing raw files...")
#         processed_data = process_all_raw_files(RAW_TEXT_DIR)

#         log_message("Validating processed data...")
#         if not processed_data:
#             raise ValueError("No valid data found after processing raw files.")
        
#         save_to_jsonl(processed_data, PREPARED_DATASET_PATH)

#         log_message("Tokenizing processed data...")
#         tokenize_dataset(PREPARED_DATASET_PATH, TOKENIZED_DATASET_PATH, MODEL_PATH)

#         log_message("Dataset preparation completed successfully.")
#     except Exception as e:
#         log_message(f"Error: {e}")
#         raise

# if __name__ == "__main__":
#     main()

























# import os
# import html2text
# import datetime
# import markdown
# import json
# from datasets import Dataset
# from llama_cpp import Llama

# # Paths
# RAW_TEXT_DIR = "./training/dataset/raw/"
# PROCESSED_TEXT_DIR = "./training/dataset/processed/"
# PREPARED_DATASET_PATH = "./training/dataset/prepared_dataset.jsonl"
# TOKENIZED_DATASET_PATH = "./training/dataset/tokenized_dataset"
# MODEL_PATH = "F:/lm-studio/models/QuantFactory/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-GGUF/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q8_0.gguf"
# LOG_FILE = "./logs/prepare_dataset.log"

# # Logging utility
# def log_message(message):
#     os.makedirs("./logs", exist_ok=True)
#     with open(LOG_FILE, "a") as log_file:
#         log_file.write(f"{datetime.datetime.now()} - {message}\n")

# # Preprocess Markdown to plain text
# def preprocess_md_to_txt(md_file, output_dir):
#     with open(md_file, "r", encoding="utf-8") as f:
#         md_content = f.read()
#     html_content = markdown.markdown(md_content)
#     plain_text = html2text.html2text(html_content).strip()
#     if not plain_text:
#         log_message(f"Warning: Markdown file '{md_file}' converted to empty text.")
#         return None
#     return plain_text

# # Process all raw files
# def process_all_raw_files(raw_dir, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     processed_texts = []
#     for filename in os.listdir(raw_dir):
#         src = os.path.join(raw_dir, filename)
#         if filename.endswith(".txt"):
#             with open(src, "r", encoding="utf-8") as f:
#                 content = f.read().strip()
#                 if content:
#                     processed_texts.append({"text": content})
#         elif filename.endswith(".md"):
#             processed_content = preprocess_md_to_txt(src, output_dir)
#             if processed_content:
#                 processed_texts.append({"text": processed_content})
#     return processed_texts

# # Save processed data as JSONL
# def save_to_jsonl(data, output_path):
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with open(output_path, "w", encoding="utf-8") as f:
#         for entry in data:
#             f.write(json.dumps(entry) + "\n")
#     log_message(f"Prepared dataset saved to '{output_path}'.")

# def tokenize_function(batch):
#     tokenized = []
#     max_length = 512  # Adjust as needed for your model

#     for text in batch["text"]:
#         try:
#             # Ensure the input is a string
#             if not isinstance(text, str):
#                 text = str(text)

#             # Log and skip empty texts
#             if not text.strip():
#                 log_message(f"Skipped empty or invalid text: {text}")
#                 continue

#             # Tokenize the text
#             tokens = Llama.tokenize(text, add_bos=True, special=False)

#             # Skip entries with zero tokens
#             if len(tokens) == 0:
#                 log_message(f"Skipped zero-length tokenization: {text[:50]}...")
#                 continue

#             # Pad or truncate tokens
#             padded = tokens[:max_length] + [0] * (max_length - len(tokens)) if len(tokens) < max_length else tokens[:max_length]
#             tokenized.append(padded)
#         except Exception as e:
#             log_message(f"Error tokenizing text: {text[:50]}... | Error: {e}")
#             continue

#     return {"input_ids": tokenized}

# def validate_dataset(prepared_dataset_path):
#     with open(prepared_dataset_path, "r", encoding="utf-8") as f:
#         data = [json.loads(line) for line in f]

#     valid_data = [entry for entry in data if entry.get("text", "").strip()]
#     if not valid_data:
#         raise ValueError("No valid entries found in the prepared dataset.")

#     log_message(f"Validated dataset: {len(valid_data)} valid entries out of {len(data)} total entries.")
#     return valid_data

# # Tokenize dataset and save
# def tokenize_dataset(jsonl_path, output_path, model_path, max_length=512):
#     log_message("Initializing Llama model for tokenization...")
#     llama = Llama(model_path=model_path)
#     try:
#         log_message("Loading prepared JSONL dataset...")
#         dataset = Dataset.from_json(jsonl_path)

#         log_message("Tokenizing dataset...")
#         def tokenize_function(batch):
#             tokenized = []
#             max_length = 512  # Set the max token length

#             for text in batch["text"]:
#                 try:
#                     # Ensure text is a string
#                     if not isinstance(text, str):
#                         text = str(text)

#                     # Skip empty or invalid texts
#                     if not text.strip():
#                         log_message(f"Skipped empty or invalid text: {text}")
#                         tokenized.append([0])  # Placeholder for empty text
#                         continue

#                     # Tokenize using Llama
#                     tokens = llama.tokenize(text, add_bos=True, special=False)

#                     # Skip entries with zero tokens
#                     if len(tokens) == 0:
#                         log_message(f"Skipped zero-length tokenization: {text[:50]}...")
#                         tokenized.append([0])  # Placeholder for zero-length tokenization
#                         continue

#                     # Pad or truncate tokens
#                     padded = tokens[:max_length] + [0] * (max_length - len(tokens)) if len(tokens) < max_length else tokens[:max_length]
#                     tokenized.append(padded)
#                 except Exception as e:
#                     log_message(f"Error tokenizing text: {text[:50]}... | Error: {e}")
#                     tokenized.append([0])  # Placeholder for errors

#             # Ensure the tokenized batch matches the input batch size
#             if len(tokenized) != len(batch["text"]):
#                 log_message(f"Batch size mismatch: input {len(batch['text'])}, tokenized {len(tokenized)}")
#                 raise ValueError("Mismatch between input and tokenized batch sizes.")

#             return {"input_ids": tokenized}
        
#         tokenized_dataset = dataset.map(tokenize_function, batched=True)

#         log_message("Validating dataset...")
#         tokenized_dataset.validate_dataset(tokenized_dataset)

#         log_message("Saving tokenized dataset to disk...")
#         tokenized_dataset.save_to_disk(output_path)
#     finally:
#         llama.close()
#         log_message("Llama model closed.")

# # Validate dataset content
# def validate_dataset(data):
#     if not data:
#         raise ValueError("No valid data found for processing.")

# # Main function
# def main():
#     try:
#         log_message("Processing raw files...")
#         processed_data = process_all_raw_files(RAW_TEXT_DIR, PROCESSED_TEXT_DIR)

#         log_message("Validating processed data...")
#         validate_dataset(processed_data)

#         log_message("Saving processed data to JSONL format...")
#         save_to_jsonl(processed_data, PREPARED_DATASET_PATH)

#         log_message("Tokenizing prepared JSONL dataset...")
#         tokenize_dataset(PREPARED_DATASET_PATH, TOKENIZED_DATASET_PATH, MODEL_PATH)

#         log_message("Dataset preparation and tokenization completed successfully.")
#     except Exception as e:
#         log_message(f"Error: {e}")
#         raise

# if __name__ == "__main__":
#     main()


# import os
# import html2text
# import datetime
# import markdown
# import json
# from datasets import Dataset

# # Paths
# RAW_TEXT_DIR = "./training/dataset/raw/"
# PROCESSED_TEXT_DIR = "./training/dataset/processed/"
# JSONL_OUTPUT_PATH = "./training/dataset/prepared_dataset.jsonl"
# LOG_FILE = "./logs/prepare_dataset.log"

# # Logging
# def log_message(message):
#     os.makedirs("./logs", exist_ok=True)
#     with open(LOG_FILE, "a") as log_file:
#         log_file.write(f"{datetime.datetime.now()} - {message}\n")

# # Preprocess Markdown to plain text
# def preprocess_md_to_txt(md_file, output_dir):
#     with open(md_file, "r", encoding="utf-8") as f:
#         md_content = f.read()
#     html_content = markdown.markdown(md_content)
#     plain_text = html2text.html2text(html_content).strip()
#     if not plain_text:
#         log_message(f"Warning: Markdown file '{md_file}' converted to empty text.")
#         return None
#     return plain_text

# # Process all raw files
# def process_all_raw_files(raw_dir, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     processed_texts = []
#     for filename in os.listdir(raw_dir):
#         src = os.path.join(raw_dir, filename)
#         if filename.endswith(".txt"):
#             with open(src, "r", encoding="utf-8") as f:
#                 content = f.read().strip()
#                 if content:
#                     processed_texts.append({"text": content})
#         elif filename.endswith(".md"):
#             processed_content = preprocess_md_to_txt(src, output_dir)
#             if processed_content:
#                 processed_texts.append({"text": processed_content})
#     return processed_texts

# # Save processed data as JSONL
# def save_to_jsonl(data, output_path):
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with open(output_path, "w", encoding="utf-8") as f:
#         for entry in data:
#             f.write(json.dumps(entry) + "\n")
#     log_message(f"Prepared dataset saved to '{output_path}'.")

# # Validate dataset directory
# def validate_dataset(data):
#     if not data:
#         raise ValueError("No valid data found for processing.")

# # Main function
# def main():
#     try:
#         log_message("Processing raw files...")
#         processed_data = process_all_raw_files(RAW_TEXT_DIR, PROCESSED_TEXT_DIR)

#         log_message("Validating processed data...")
#         validate_dataset(processed_data)

#         log_message("Saving processed data to JSONL format...")
#         save_to_jsonl(processed_data, JSONL_OUTPUT_PATH)

#         log_message("Dataset preparation completed successfully.")
#     except Exception as e:
#         log_message(f"Error: {e}")
#         raise

# if __name__ == "__main__":
#     main()



# import os
# import html2text
# import datetime
# import markdown
# import re
# from datasets import Dataset
# from llama_cpp import Llama

# RAW_TEXT_DIR = "./training/dataset/raw/"
# PROCESSED_TEXT_DIR = "./training/dataset/processed/"
# TOKENIZED_DATASET_PATH = "./training/dataset/tokenized_dataset"
# MODEL_PATH = "F:/lm-studio/models/QuantFactory/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-GGUF/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q8_0.gguf"
# LOG_FILE = "./logs/prepare_dataset.log"

# def log_message(message):
#     if not os.path.exists("./logs"):
#         os.makedirs("./logs")
#     with open(LOG_FILE, "a") as log_file:
#         log_file.write(f"{datetime.datetime.now()} - {message}\n")

# def preprocess_md_to_txt(md_file, output_dir):
#     with open(md_file, "r", encoding="utf-8") as f:
#         md_content = f.read()
#     html_content = markdown.markdown(md_content)
#     plain_text = html2text.html2text(html_content).strip()
#     if not plain_text:
#         log_message(f"Warning: Markdown file '{md_file}' converted to empty text.")
#         return
#     base_name = os.path.basename(md_file).replace(".md", ".txt")
#     output_path = os.path.join(output_dir, base_name)
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(plain_text)

# def process_all_raw_files(raw_dir, output_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     for filename in os.listdir(raw_dir):
#         src = os.path.join(raw_dir, filename)
#         if filename.endswith(".txt"):
#             with open(src, "r", encoding="utf-8") as f:
#                 content = f.read()
#             dst = os.path.join(output_dir, filename)
#             with open(dst, "w", encoding="utf-8") as f:
#                 f.write(content)
#         elif filename.endswith(".md"):
#             preprocess_md_to_txt(src, output_dir)

# def validate_dataset_path(path):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Dataset path '{path}' does not exist.")
#     if not any(fname.endswith(".txt") for fname in os.listdir(path)):
#         raise ValueError(f"No text files found in '{path}'.")

# def load_raw_data(path):
#     texts = []
#     for filename in os.listdir(path):
#         if filename.endswith(".txt"):
#             with open(os.path.join(path, filename), "r", encoding="utf-8") as file:
#                 txt = file.read().strip()
#                 if txt:
#                     texts.append(txt)
#     return texts

# def save_tokenized_dataset(dataset, output_path):
#     if not os.path.exists(os.path.dirname(output_path)):
#         os.makedirs(os.path.dirname(output_path))
#     dataset.save_to_disk(output_path)
#     log_message(f"Tokenized dataset saved to '{output_path}'.")

# def clean_text(text):
#     text = text.encode("ascii", errors="ignore").decode("ascii")
#     # Also remove braces/brackets if possible:
#     text = re.sub(r"[{}[\]]", "", text)
#     return text.strip()

# def tokenize_function(batch, llama, max_length=512):
#     tokenized = []
#     for text in batch["text"]:
#         text = clean_text(text)
#         try:
#             tokens = llama.tokenize(text, add_bos=True, special=False)
#         except:
#             tokens = []
#         if not tokens:
#             tokens = [0]
#         tokens = tokens[:max_length] + [0]*(max_length - len(tokens))
#         tokenized.append(tokens)
#     return {"input_ids": tokenized}

# # def tokenize_function(batch, llama, max_length=512):
# #     tokenized = []
# #     # Produce exactly one tokenized result per string to avoid mismatches
# #     for text in batch["text"]:
# #         text = str(text).encode("utf-8", errors="ignore").decode("utf-8")
# #         try:
# #             tokens = llama.tokenize(text, add_bos=True, special=False)
# #         except Exception as e:
# #             log_message(f"Error tokenizing text: {text[:50]}... | Error: {e}")
# #             tokens = [0]
# #         if not tokens:
# #             tokens = [0]
# #         if len(tokens) < max_length:
# #             tokens += [0] * (max_length - len(tokens))
# #         else:
# #             tokens = tokens[:max_length]
# #         tokenized.append(tokens)
# #     return {"input_ids": tokenized}

# def main():
#     llama = None
#     try:
#         log_message("Initializing Llama model...")
#         llama = Llama(model_path=MODEL_PATH)

#         log_message("Processing raw files...")
#         process_all_raw_files(RAW_TEXT_DIR, PROCESSED_TEXT_DIR)

#         validate_dataset_path(PROCESSED_TEXT_DIR)

#         log_message("Loading processed text data...")
#         raw_texts = load_raw_data(PROCESSED_TEXT_DIR)
#         if not raw_texts:
#             raise ValueError("All processed files are empty or invalid.")

#         dataset = Dataset.from_dict({"text": raw_texts})

#         log_message("Tokenizing dataset...")
#         def fn(batch):
#             return tokenize_function(batch, llama)

#         tokenized_dataset = dataset.map(fn, batched=True)

#         if len(tokenized_dataset) == 0:
#             raise ValueError("Tokenization resulted in an empty dataset.")

#         log_message("Saving tokenized dataset...")
#         save_tokenized_dataset(tokenized_dataset, TOKENIZED_DATASET_PATH)
#         log_message("Dataset preparation completed successfully.")
#     finally:
#         if llama is not None:
#             log_message("Closing Llama model...")
#             llama.close()

# if __name__ == "__main__":
#     main()




# # File: training/prepare_dataset.py
# # Purpose: Processes raw text and Markdown files, aggregates data, and prepares a tokenized dataset for training.

# import os
# import html2text
# import datetime
# import markdown
# from datasets import Dataset
# from llama_cpp import Llama

# # Define dataset paths and model
# RAW_TEXT_DIR = "./training/dataset/raw/"  # Directory containing raw text and Markdown files
# PROCESSED_TEXT_DIR = "./training/dataset/processed/"  # Directory for processed text files
# TOKENIZED_DATASET_PATH = "./training/dataset/tokenized_dataset"  # Output path
# MODEL_PATH = "F:/lm-studio/models/QuantFactory/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-GGUF/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q8_0.gguf"
# LOG_FILE = "./logs/prepare_dataset.log"  # Log file for debugging

# def log_message(message):
#     """Log messages to a file."""
#     if not os.path.exists("./logs"):
#         os.makedirs("./logs")
#     with open(LOG_FILE, "a") as log_file:
#         log_file.write(f"{datetime.datetime.now()} - {message}\n")

# def preprocess_md_to_txt(md_file, output_dir):
#     """Converts Markdown files to plain text and saves them as .txt."""
#     with open(md_file, "r", encoding="utf-8") as f:
#         md_content = f.read()

#     # Convert Markdown to plain text
#     plain_text = markdown.markdown(md_content)

#     # Save the processed text to a .txt file
#     base_name = os.path.basename(md_file).replace(".md", ".txt")
#     output_path = os.path.join(output_dir, base_name)
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(plain_text)

#     print(f"Processed and saved: {output_path}")

# def process_all_raw_files(raw_dir, output_dir):
#     """Processes all raw text and Markdown files in the raw dataset directory."""
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for filename in os.listdir(raw_dir):
#         file_path = os.path.join(raw_dir, filename)
#         if filename.endswith(".txt"):
#             # Copy text files directly
#             with open(file_path, "r", encoding="utf-8") as f:
#                 content = f.read()
#             output_path = os.path.join(output_dir, filename)
#             with open(output_path, "w", encoding="utf-8") as f:
#                 f.write(content)
#             print(f"Copied text file: {output_path}")
#         elif filename.endswith(".md"):
#             # Convert Markdown files to text
#             preprocess_md_to_txt(file_path, output_dir)

# def validate_dataset_path(processed_dir):
#     """Validate that the processed dataset path exists and contains text files."""
#     if not os.path.exists(processed_dir):
#         log_message(f"Error: Dataset path '{processed_dir}' does not exist.")
#         raise FileNotFoundError(f"Dataset path '{processed_dir}' does not exist.")
#     if not any(fname.endswith(".txt") for fname in os.listdir(processed_dir)):
#         log_message(f"Error: No text files found in '{processed_dir}'.")
#         raise ValueError(f"No text files found in '{processed_dir}'.")

# def load_raw_data(processed_dir):
#     """Load processed text files from the directory."""
#     texts = []
#     for filename in os.listdir(processed_dir):
#         if filename.endswith(".txt"):
#             with open(os.path.join(processed_dir, filename), "r", encoding="utf-8") as file:
#                 texts.append(file.read())
#     return texts

# def save_tokenized_dataset(dataset, output_path):
#     """Save the dataset to disk."""
#     if not os.path.exists(os.path.dirname(output_path)):
#         os.makedirs(os.path.dirname(output_path))
#     dataset.save_to_disk(output_path)
#     log_message(f"Tokenized dataset saved to '{output_path}'.")

# def preprocess_md_to_txt(md_file, output_dir):
#     """Converts Markdown files to plain text and saves them as .txt."""
#     with open(md_file, "r", encoding="utf-8") as f:
#         md_content = f.read()

#     # Convert Markdown to plain text
#     html_content = markdown.markdown(md_content)
#     plain_text = html2text.html2text(html_content).strip()  # Remove residual HTML tags and extra spaces

#     if not plain_text:
#         log_message(f"Warning: Markdown file '{md_file}' converted to empty text.")
#         return

#     # Save the processed text to a .txt file
#     base_name = os.path.basename(md_file).replace(".md", ".txt")
#     output_path = os.path.join(output_dir, base_name)
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(plain_text)

#     print(f"Processed and saved: {output_path}")

# def tokenize_function(batch):
#     """Tokenizes a batch of texts."""
#     tokenized = []
#     max_length = 512

#     for text in batch["text"]:
#         try:
#             if not text.strip():
#                 log_message(f"Skipped empty or invalid text: {text}")
#                 continue

#             # Tokenize the text
#             tokens = llama.tokenize(text, add_bos=True, special=False)

#             if len(tokens) == 0:
#                 log_message(f"Skipped zero-length tokenization: {text[:50]}...")
#                 continue

#             # Pad or truncate tokens
#             padded = tokens[:max_length] + [0] * (max_length - len(tokens)) if len(tokens) < max_length else tokens[:max_length]
#             tokenized.append(padded)
#         except Exception as e:
#             log_message(f"Error tokenizing text: {text[:50]}... | Error: {e}")
#             continue

#     if not tokenized:
#         log_message("Warning: No valid tokenized entries found in this batch.")

#     return {"input_ids": tokenized}


# def main():
#     llama = None  # Ensure we can explicitly close the Llama instance
#     try:
#         # Initialize Llama instance
#         log_message("Initializing Llama model...")
#         llama = Llama(model_path=MODEL_PATH)

#         # Process raw files
#         log_message("Processing raw files...")
#         process_all_raw_files(RAW_TEXT_DIR, PROCESSED_TEXT_DIR)

#         # Validate processed dataset path
#         validate_dataset_path(PROCESSED_TEXT_DIR)

#         # Load processed text data
#         log_message("Loading processed text data...")
#         raw_texts = load_raw_data(PROCESSED_TEXT_DIR)

#         # Filter out empty texts
#         raw_texts = [text for text in raw_texts if text.strip()]
#         if not raw_texts:
#             raise ValueError("All processed files are empty or invalid.")

#         # Create Hugging Face dataset
#         log_message("Creating Hugging Face dataset...")
#         dataset = Dataset.from_dict({"text": raw_texts})

#         # Tokenize the dataset
#         log_message("Tokenizing dataset...")
#         def tokenize_function(batch):
#             tokenized = []
#             max_length = 512
#             for text in batch["text"]:
#                 try:
#                     # Skip empty or invalid text
#                     if not text.strip():
#                         log_message(f"Skipped empty or invalid text: {text}")
#                         continue

#                     # Tokenize the text
#                     tokens = llama.tokenize(text, add_bos=True, special=False)

#                     # Skip entries with zero tokens
#                     if len(tokens) == 0:
#                         log_message(f"Skipped zero-length tokenization: {text[:50]}...")
#                         continue

#                     # Pad or truncate tokens
#                     padded = tokens[:max_length] + [0] * (max_length - len(tokens)) if len(tokens) < max_length else tokens[:max_length]
#                     tokenized.append(padded)
#                 except Exception as e:
#                     log_message(f"Error tokenizing text: {text[:50]}... | Error: {e}")
#                     continue
#             if len(tokenized) != len(batch["text"]):
#                 raise ValueError("Mismatch in tokenized outputs and input batch sizes.")
#             return {"input_ids": tokenized}

#         tokenized_dataset = dataset.map(tokenize_function, batched=True)

#         # Check if tokenized dataset is empty
#         if len(tokenized_dataset) == 0:
#             raise ValueError("Tokenization resulted in an empty dataset. Check input files.")

#         # Save tokenized dataset to disk
#         log_message("Saving tokenized dataset...")
#         save_tokenized_dataset(tokenized_dataset, TOKENIZED_DATASET_PATH)

#         log_message("Dataset preparation completed successfully.")
#     except Exception as e:
#         log_message(f"Error occurred: {e}")
#         raise
#     finally:
#         # Ensure llama instance is closed properly
#         if llama is not None:
#             log_message("Closing Llama model...")
#             llama.close()

# if __name__ == "__main__":
#     main()









# # # File: training/prepare_dataset.py
# # # Purpose: Prepares a tokenized dataset for evaluation.

# # import os
# # from datasets import Dataset
# # import datetime
# # from llama_cpp import Llama

# # # Define paths
# # RAW_TEXT_DIR = "./training/dataset/raw/"
# # TOKENIZED_DATASET_PATH = "./training/dataset/tokenized_dataset"
# # MODEL_PATH = "F:/lm-studio/models/QuantFactory/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-GGUF/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q8_0.gguf"
# # LOG_FILE = "./logs/prepare_dataset.log"

# # def log_message(message):
# #     """Log messages to a file."""
# #     if not os.path.exists("./logs"):
# #         os.makedirs("./logs")
# #     with open(LOG_FILE, "a") as log_file:
# #         log_file.write(f"{datetime.datetime.now()} - {message}\n")

# # def validate_dataset_path(raw_text_dir):
# #     """Validate that the dataset path exists and contains text files."""
# #     if not os.path.exists(raw_text_dir):
# #         log_message(f"Error: Dataset path '{raw_text_dir}' does not exist.")
# #         raise FileNotFoundError(f"Dataset path '{raw_text_dir}' does not exist.")
# #     if not any(fname.endswith(".txt") for fname in os.listdir(raw_text_dir)):
# #         log_message(f"Error: No text files found in '{raw_text_dir}'.")
# #         raise ValueError(f"No text files found in '{raw_text_dir}'.")

# # def load_raw_data(raw_text_dir):
# #     """Load raw text files from the directory."""
# #     texts = []
# #     for filename in os.listdir(raw_text_dir):
# #         if filename.endswith(".txt"):
# #             with open(os.path.join(raw_text_dir, filename), "r", encoding="utf-8") as file:
# #                 texts.append(file.read())
# #     return texts

# # def save_tokenized_dataset(dataset, output_path):
# #     """Save the dataset to disk."""
# #     if not os.path.exists(os.path.dirname(output_path)):
# #         os.makedirs(os.path.dirname(output_path))
# #     dataset.save_to_disk(output_path)
# #     log_message(f"Tokenized dataset saved to '{output_path}'.")

# # def main():
# #     # Initialize Llama instance
# #     llama = Llama(model_path=MODEL_PATH)

# #     try:
# #         # Validate dataset path
# #         validate_dataset_path(RAW_TEXT_DIR)

# #         # Load raw text data
# #         log_message("Loading raw text data...")
# #         raw_texts = load_raw_data(RAW_TEXT_DIR)

# #         # Create Hugging Face dataset
# #         log_message("Creating Hugging Face dataset...")
# #         dataset = Dataset.from_dict({"text": raw_texts})

# #         def tokenize_function(batch):
# #             tokenized = []
# #             for text in batch["text"]:
# #                 # Ensure the text is encoded as plain UTF-8
# #                 if not isinstance(text, str):
# #                     raise ValueError(f"Invalid input type: {type(text)}. Expected a string.")
# #                 text = text.encode("utf-8")  # Ensure bytes format for compatibility
# #                 tokens = llama.tokenize(text, add_bos=True, special=False)
# #                 tokenized.append(tokens)

# #             max_length = 512
# #             padded = [
# #                 tokens[:max_length] + [0] * (max_length - len(tokens))
# #                 if len(tokens) < max_length
# #                 else tokens[:max_length]
# #                 for tokens in tokenized
# #             ]
# #             return {"input_ids": padded}

# #         # Tokenize the dataset
# #         log_message("Tokenizing dataset...")
# #         tokenized_dataset = dataset.map(tokenize_function, batched=True)

# #         # Save tokenized dataset to disk
# #         log_message("Saving tokenized dataset...")
# #         save_tokenized_dataset(tokenized_dataset, TOKENIZED_DATASET_PATH)

# #         log_message("Dataset preparation completed successfully.")
# #     finally:
# #         # Explicitly close the Llama instance
# #         llama.close()

# # if __name__ == "__main__":
# #     main()
