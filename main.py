# File: main.py
# Purpose: Main entry point for the autonomous LLM application.

import queue
import threading
import datetime
from llm.autonomous_llm import AutonomousLLM
from llm.tts_module import TextToSpeech
from llm.wake_word import WakeWordDetector


def main():
    # Initialize components
    llm = AutonomousLLM()
    tts = TextToSpeech()
    command_queue = queue.Queue()

    # Start the wake-word detector in a separate thread
    wake_detector = WakeWordDetector(wake_word="wake up")
    wake_thread = threading.Thread(
        target=wake_detector.listen_for_wake_word, args=(command_queue,)
    )
    wake_thread.daemon = True
    wake_thread.start()

    print("[System] Waiting for wake word to activate...")

    try:
        while True:
            if not command_queue.empty():
                command = command_queue.get()
                if command == "wake":
                    print("[System] Wake word triggered! Generating response...")

                    # Generate thoughts
                    try:
                        thoughts = llm.generate_inner_thoughts()
                        print("Generated Thought:", thoughts)

                        # Speak the response if appropriate
                        if "speak now" in thoughts.lower():
                            print("[System] Speaking generated thought...")
                            tts.speak(thoughts)

                        # Retrieve and display recent thoughts
                        recent_thoughts = llm.recall_recent_thoughts()
                        print("\n[Memory Recall - Recent Thoughts]")
                        for idx, thought in enumerate(recent_thoughts, 1):
                            print(f"{idx}: {thought}")

                    except Exception as e:
                        print(f"[Error] Failed to process thought: {e}")

    except KeyboardInterrupt:
        print("\n[System] Shutting down gracefully...")
    except Exception as e:
        print(f"[System Error] Unexpected error: {e}")


if __name__ == "__main__":
    main()