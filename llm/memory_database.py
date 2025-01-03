# File: llm/memory_database.py
# Purpose: Provides database logic for saving and retrieving thoughts to enable persistent memory.

import os
import sqlite3
import datetime
import torch

# SQLite Database logic for offline memory
class MemoryDatabase:
    def __init__(self, db_path="memory.db"):
        self.db_path = db_path
        self.initialize_db()  # Changed from _initialize_db()
        
    def initialize_db(self):
        """Initialize the database with necessary tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def save_memory(self, key, value):
        """Save or update a memory."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO memories (key, value, timestamp)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (key, value))
            conn.commit()
            
    def get_memory(self, key):
        """Retrieve a memory by key."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM memories WHERE key = ?", (key,))
            result = cursor.fetchone()
            return result[0] if result else None
            
    def get_all_memories(self):
        """Retrieve all memories."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value, timestamp FROM memories ORDER BY timestamp DESC")
            return cursor.fetchall()

    def save_thought(self, thought):
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO memory (timestamp, thought) VALUES (?, ?)",
                (datetime.datetime.now().isoformat(), thought)
            )
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            conn.close()

    def retrieve_recent_thoughts(self, limit=10):
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT thought FROM memory ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            thoughts = [row[0] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            thoughts = []
        finally:
            conn.close()
        return thoughts
    
    def store_context(self, context):
        """Store a new context in memory."""
        self.memory.append(context)

    def retrieve_context(self):
        """Retrieve the most recent context."""
        if self.memory:
            return self.memory[-1]
        return None

# # Update AutonomousLLM class to include memory handling
# class AutonomousLLM:
#     def __init__(self, model_name=MODEL_NAME, max_len=512, memory_db_path="llm_memory.db"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype=torch.float16,
#             device_map="auto"
#         )
#         self.max_len = max_len
#         self.system_prompt = (
#             "You have the capacity to generate your own thoughts. "
#             "Maintain hidden reasoning and produce textual output, "
#             "even when not explicitly prompted by a user."
#         )
#         self.hidden_context = self.system_prompt
#         self.memory = MemoryDatabase(memory_db_path)

#     def generate_inner_thoughts(self):
#         inputs = self.tokenizer.encode(
#             self.hidden_context, return_tensors='pt'
#         ).to(self.model.device)

#         output_tokens = self.model.generate(
#             inputs,
#             max_length=len(inputs[0]) + 64,
#             temperature=0.8,
#             do_sample=True,
#             top_p=0.9,
#             pad_token_id=self.tokenizer.eos_token_id
#         )

#         new_text = self.tokenizer.decode(
#             output_tokens[0][len(inputs[0]):], skip_special_tokens=True
#         )

#         # Update hidden context and save to memory
#         self.hidden_context += new_text
#         self.memory.save_thought(new_text)
#         return new_text

#     def recall_recent_thoughts(self):
#         return self.memory.retrieve_recent_thoughts()

# # Update main loop to include memory recall
# def main_loop():
#     llm = AutonomousLLM()
#     while True:
#         # Generate thoughts and save to memory
#         thoughts = llm.generate_inner_thoughts()
#         print("Generated Thought:", thoughts)

#         # Retrieve and print recent thoughts periodically
#         recent_thoughts = llm.recall_recent_thoughts()
#         print("\n[Memory Recall - Recent Thoughts]")
#         for idx, thought in enumerate(recent_thoughts, 1):
#             print(f"{idx}: {thought}")

#         # Decide whether to trigger speech
#         if "speak now" in thoughts.lower():
#             print("Autonomous Speech Output:", thoughts)

# if __name__ == "__main__":
#     main_loop()