# Local RAG v2 🔍

A fully local, open-source Retrieval-Augmented Generation (RAG) search engine for your personal documents. No data leaves your machine, no APIs required, and it runs efficiently even on an 8GB RAM machine without a dedicated GPU.

## Features

- **Fully Private & Offline:** All processing, embedding, and answering happens on your local device.
- **Efficient Local Models:** Uses `all-MiniLM-L6-v2` for lightweight embeddings and `qwen2:1.5b` (via Ollama) for fast, local LLM answers.
- **Built-in Vector DB:** Embedded ChromaDB persists data locally without requiring a separate server process.
- **Smart Chunking & Deduplication:** MD5 hashing prevents re-indexing, and sliding window chunking ensures high-quality search retrieval.
- **User-friendly UI:** Streamlit-based web interface with folder pickers, live progress bars, a kill switch for indexing, and high-quality search result highlights.
- **Supports Multiple Formats:** Easily index `.pdf` (via PyMuPDF), `.docx` (via python-docx), and `.txt` files.

## Prerequisites

1. **Python 3.10+** (Ensure it is added to your PATH)
2. **Ollama** — Download and install from [https://ollama.com/download](https://ollama.com/download)

## Setup & Running the Application

### 1. Start Ollama and Pull the Model
First, ensure the Ollama server is running. You can start it from your system tray or run:
```powershell
ollama serve
```

Open a new terminal and pull the required LLM (this is a one-time ~1GB download):
```powershell
ollama pull qwen2:1.5b
```

### 2. Install Python Dependencies
Create and activate a virtual environment (recommended):
```powershell
python -m venv venv
.\venv\Scripts\activate
```

Install the required packages:
```powershell
pip install -r requirements.txt
```

### 3. Start the Application
Run the Streamlit application:
```powershell
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

## Usage

1. **Select Folders:** Use the sidebar to pick which folders (Documents, Downloads, Desktop, etc.) or enter custom paths to index.
2. **Index Files:** Click the **Start Indexing** button. You can monitor the progress or stop it at any time.
3. **Search:** Enter a natural language query in the main search bar to find relevant documents.
4. **Get Answers:** The local LLM will generate an answer based purely on the retrieved context, and the source documents will be displayed below with confidence scores.

## Troubleshooting

| Issue | Solution |
|---|---|
| "Cannot connect to Ollama" | Make sure the Ollama app is running or execute `ollama serve` in a terminal. |
| Slow indexing | Normal for CPU processing. Monitor the progress bar or index smaller folders first. |
| Out of memory / Crashing | Try closing other applications or index fewer files at a time. |
| No search results | Try lowering the relevance threshold slider in the sidebar. |

## Upgrading the LLM

If your machine has more RAM available, you can upgrade the LLM for better answers. Simply pull a larger model:
```powershell
ollama pull phi3:mini
# or
ollama pull llama3.2
```
Then, update the `LLM_MODEL` variable inside `rag_engine.py` to match the new model name and restart the app.
