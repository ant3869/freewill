# File: inspect_dataset.py
import json

# Path to the prepared JSONL dataset
PREPARED_DATASET_PATH = "./training/dataset/prepared_dataset.jsonl"

def inspect_dataset(file_path, limit=5):
    """
    Inspect the prepared dataset by printing the first few entries.

    Args:
        file_path (str): Path to the JSONL dataset.
        limit (int): Number of entries to display.
    """
    try:
        print(f"Inspecting dataset: {file_path}\n")
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                entry = json.loads(line)
                print(f"Entry {i}: {json.dumps(entry, indent=4)}\n")
                if i + 1 >= limit:  # Stop after printing the specified limit
                    break
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON on line {i + 1}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    inspect_dataset(PREPARED_DATASET_PATH)
