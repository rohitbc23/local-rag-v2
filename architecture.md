# Local RAG v2 — Architecture

## System Architecture

```mermaid
graph TD
    User([👤 User]) -->|"Browser"| UI

    subgraph "Frontend — Streamlit (app.py)"
        UI["🖥️ Streamlit App"]
        FP["📁 Folder Picker<br/>Checkboxes + Custom Input"]
        SB["🔍 Search Bar"]
        PB["📊 Progress Bar<br/>Real-time % + file counts"]
        RC["📄 Result Cards<br/>Confidence scores, snippets"]
        UI --- FP
        UI --- SB
        UI --- PB
        UI --- RC
    end

    subgraph "Core Engine — Python (rag_engine.py)"
        Engine["⚙️ LocalRAG Class"]
        Parser["📄 File Parser<br/>PDF (PyMuPDF) · DOCX · TXT"]
        Chunker["✂️ Sliding Window Chunker<br/>500 chars, 150 overlap"]
        Embedder["🧠 all-MiniLM-L6-v2<br/>22MB, 384-dim vectors"]
        LLM["🤖 Ollama → qwen2:1.5b<br/>1.5B params, ~1GB RAM"]
        Engine --> Parser --> Chunker --> Embedder
        Engine --> LLM
    end

    subgraph "Storage — Local Disk Only"
        VectorDB[("💾 ChromaDB<br/>SQLite-backed<br/>./local_rag_db")]
        Files["📂 User's Files<br/>(PDF, DOCX, TXT)"]
    end

    FP -->|"Selected folders"| Engine
    SB -->|"Query"| Engine
    Embedder -->|"Vectors"| VectorDB
    VectorDB -->|"Top-K chunks"| LLM
    LLM -->|"Answer"| RC
    Engine -->|"Reads"| Files

    style UI fill:#667eea,stroke:#764ba2,color:white
    style Engine fill:#0f3460,stroke:#16213e,color:#e94560
    style VectorDB fill:#533483,stroke:#16213e,color:white
```

## Data Flow

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant App as Streamlit UI
    participant E as RAG Engine
    participant Emb as MiniLM-L6-v2
    participant DB as ChromaDB
    participant LLM as Ollama (qwen2:1.5b)

    Note over U,LLM: ═══ INDEXING FLOW ═══
    U->>App: ☑️ Select folders via checkboxes
    U->>App: ▶️ Click "Start Indexing"
    App->>E: scan_and_index(folders)
    loop For each file
        E->>E: Parse PDF/DOCX/TXT → text
        E->>E: Chunk (500 chars, 150 overlap)
        E->>Emb: Encode chunks → 384-dim vectors
        Emb->>DB: Upsert vectors + metadata
        E->>App: Progress callback (percent, filename)
        App->>U: Update progress bar + stats
    end

    Note over U,LLM: ═══ SEARCH FLOW ═══
    U->>App: Type "invoice from October"
    App->>E: search(query, threshold=0.45)
    E->>Emb: Encode query → 384-dim vector
    Emb->>DB: Cosine similarity search
    DB->>E: Top-K matching chunks
    E->>LLM: "Answer using this context: ..."
    LLM->>E: Generated answer
    E->>App: Answer + source documents
    App->>U: AI answer box + result cards
```

## Component Table

| Component | Technology | RAM Usage | License | Why This Choice |
|---|---|---|---|---|
| **Embedding** | `all-MiniLM-L6-v2` | ~80MB | Apache 2.0 | 22MB model, 384-dim. 50% less memory than mpnet. Only 1.5% lower accuracy. |
| **Vector DB** | ChromaDB (embedded) | ~50MB | Apache 2.0 | No server process. SQLite persistence. Zero config. |
| **Local LLM** | Ollama → `qwen2:1.5b` | ~1GB | MIT / Apache 2.0 | Fits in 8GB RAM. Good instruction-following for 1.5B model. |
| **UI** | Streamlit | ~100MB | Apache 2.0 | Pure Python. Built-in widgets. No frontend build step. |
| **PDF Parser** | PyMuPDF (fitz) | Minimal | AGPL-3.0 | 10x faster than PyPDF2. Low memory. Handles complex layouts. |
| **DOCX Parser** | python-docx | Minimal | MIT | Standard, reliable, lightweight. |
| **Total** | | **~1.3GB** | | Leaves ~6.7GB for OS on 8GB machine |

## Chunking Strategy

```
Document: "The quick brown fox jumps over the lazy dog. The dog barked..."
                                                                          
Chunk 1: |←————————— 500 chars ————————→|                                
Chunk 2:              |←————150 overlap——|←—— 350 new ——→|              
Chunk 3:                                  |←—— 150 ——|←—— 350 ——→|      

WHY: 30% overlap ensures no sentence is lost at boundaries.
     500 chars gives precise search hits (vs 1000 = too broad).
```
