<<<<<<< HEAD
# **Autonomous LLM**
An offline, open-source large language model (LLM) system designed for autonomous reasoning, free speech generation, and inner thought processing. This project integrates modules for wake-word detection, text-to-speech synthesis, and persistent memory using SQLite.

---

## **Features**
- **Inner Thought Mechanism:** Enables the LLM to generate reasoning or thoughts autonomously.
- **Wake-Word Detection:** Responds to a user-defined wake word.
- **Text-to-Speech (TTS):** Converts generated text to speech using Coqui TTS.
- **Persistent Memory:** Stores and recalls generated thoughts using SQLite.
- **Fine-Tuning Support:** Tools for preparing datasets and fine-tuning the LLM.

---

## **Project Structure**
```plaintext
autonomous_llm/
├── main.py                  # Entry point for the application
├── llm/
│   ├── autonomous_llm.py    # Core logic for the LLM
│   ├── memory_database.py   # SQLite memory logic
│   ├── tts_module.py        # Text-to-Speech wrapper
│   ├── wake_word.py         # Wake-word detection logic
├── training/
│   ├── prepare_dataset.py   # Converts raw text to tokenized dataset
│   ├── fine_tune.py         # Fine-tunes the LLM model
│   ├── dataset/             # Directory for raw and tokenized datasets
├── logs/                    # Runtime and training logs
└── requirements.txt         # List of dependencies
```

---

## **Setup Instructions**
### **1. Clone the Repository**
```bash
git clone https://github.com/your-repo-name/autonomous-llm.git
cd autonomous-llm
```

### **2. Install Dependencies**
Install all required packages using `pip`:
```bash
pip install -r requirements.txt
```

**Note:** For PyTorch, install the appropriate version for your system. For example:
- **With CUDA support:**
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- **CPU-only:**
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

### **3. Prepare Your Dataset**
1. Place your raw `.txt` files in the `training/dataset/raw/` directory.
2. Run the dataset preparation script:
   ```bash
   python training/prepare_dataset.py
   ```
3. The tokenized dataset will be saved to `training/dataset/tokenized_dataset`.

### **4. Fine-Tune the LLM**
To fine-tune the LLM:
```bash
python training/fine_tune.py
```
The fine-tuned model will be saved in the `fine_tuned_model/` directory.

---

## **Running the Application**
Start the application by running:
```bash
python main.py
```

### **Main Features:**
- **Wake Word:** Activates on the wake word "hello assistant" (can be configured in `wake_word.py`).
- **Autonomous Thoughts:** The LLM generates and stores inner thoughts.
- **Speech Output:** Generated thoughts tagged with "speak now" are converted to speech.

---

## **Dependencies**
Below are the required Python packages:
```plaintext
transformers==4.31.0
datasets==2.14.5
TTS==0.10.0
SpeechRecognition==3.8.1
webrtcvad==2.0.10
numpy==1.24.3
scipy==1.11.1
torch==2.0.1
sqlite3 (Standard library; no installation required)
```

Install them using:
```bash
pip install -r requirements.txt
```

---

## **Known Issues**
1. **Audio Playback Issues:**
   - Ensure the required audio playback tools are installed (`aplay` on Linux or `mplay32` on Windows).
2. **Performance:**
   - Running on CPU may result in slower performance. A GPU is recommended for heavy workloads.
3. **Wake-Word Accuracy:**
   - Performance in noisy environments can be improved by tweaking the `wake_word.py` filter settings.

---

## **Contributing**
Contributions are welcome! Please fork the repository and submit a pull request.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for more information.

---

## **Contact**
For questions or support, please contact:
- **Author:** Anthony
- **Email:** your-email@example.com
=======
# freewill
autonomous open source llm solution experiment
>>>>>>> a1745a944a75fcb337669cc8b0f60041c45d87c0
