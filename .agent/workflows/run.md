---
description: How to run the Local RAG v2 application
---

# Running Local RAG v2

## Prerequisites

1. **Python 3.10+** — must be installed and on PATH
2. **Ollama** — download and install from https://ollama.com/download

## Steps

### 1. Make sure Ollama is running
// turbo
```powershell
ollama serve
```
> If Ollama is already running (e.g. via system tray), this will report "address already in use" — that's fine.

### 2. Pull the LLM model (one-time, ~1GB download)
```powershell
ollama pull qwen2:1.5b
```

### 3. Activate the virtual environment
// turbo
```powershell
c:\Users\bcroh\.gemini\antigravity\scratch\local_rag_v2\venv\Scripts\activate
```

### 4. Install dependencies (if not already installed)
```powershell
pip install -r c:\Users\bcroh\.gemini\antigravity\scratch\local_rag_v2\requirements.txt
```

### 5. Run the Streamlit app
```powershell
streamlit run c:\Users\bcroh\.gemini\antigravity\scratch\local_rag_v2\app.py
```
> The app opens in your browser at http://localhost:8501

## Usage (in the browser)

1. **Select folders** in the sidebar (Documents, Downloads, etc.)
2. Click **▶️ Start Indexing** to scan and process your files
3. **Type a query** in the search bar to find relevant documents
4. View the **AI-generated answer** and **source documents**

## Troubleshooting

| Issue | Fix |
|---|---|
| "Cannot connect to Ollama" | Run `ollama serve` in a separate terminal |
| Slow indexing | Normal on CPU — progress bar shows % complete |
| Out of memory | Index smaller folders first, close other apps |
| No search results | Lower the relevance threshold slider in the sidebar |

## Upgrading the LLM

If you have more RAM, swap to a better model:

```powershell
ollama pull phi3:mini       # 3.8B params, ~2.3GB RAM
# or
ollama pull llama3.2        # 3B params, ~2GB RAM
```

Then update `LLM_MODEL` in `rag_engine.py`.
