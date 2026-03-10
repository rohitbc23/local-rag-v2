"""
rag_engine.py — Core RAG (Retrieval-Augmented Generation) Engine
================================================================

DESIGN DECISIONS (read this first):
------------------------------------
1. EMBEDDING MODEL: all-MiniLM-L6-v2 (22MB, 384-dim vectors)
   - WHY: On 8GB RAM without GPU, every MB matters. This model uses ~80MB RAM
     at runtime vs ~400MB for larger models like all-mpnet-base-v2.
   - TRADEOFF: 1.5% lower accuracy on benchmarks (68.1 vs 69.6 STS score),
     which is negligible for local file search.
   - LICENSE: Apache 2.0 (fully open-source)

2. VECTOR DATABASE: ChromaDB (embedded, SQLite-backed)
   - WHY: Runs in-process (no separate server), zero configuration, persists
     to disk automatically. Alternatives like Qdrant/Weaviate require a 
     separate server process eating more RAM.
   - LICENSE: Apache 2.0

3. LOCAL LLM: Ollama with qwen2:1.5b
   - WHY: At 1.5B parameters (~1GB RAM), it leaves plenty of headroom on 8GB.
     Starting small allows us to validate the pipeline, then scale up to 
     phi3:mini (3.8B) or higher if RAM allows.
   - LICENSE: Ollama (MIT), Qwen2 (Apache 2.0)

4. CHUNKING: 500 chars with 150 char overlap
   - WHY: Smaller chunks (vs the typical 1000) give more precise search hits.
     150 overlap (30%) ensures sentences split across boundaries are captured.
     We sacrifice some speed (more chunks = more embeddings) for accuracy.
   - TARGET: ~55% search relevance accuracy, tunable via threshold.

5. FILE HASHING: MD5 per file for deduplication
   - WHY: Prevents re-indexing unchanged files. MD5 is fast and sufficient
     for content-change detection (not security).

Author: Local RAG v2
"""

import os
import sys
import gc
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Generator, Callable, Tuple

import chromadb
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF — fastest Python PDF parser, low memory footprint
from docx import Document
import ollama


# ---------------------------------------------------------------------------
# Logging setup — replaces the old Traceloop/Dynatrace cloud dependency.
# Everything stays local. Logs go to a file + console.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("rag_debug.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("LocalRAG")


class LocalRAG:
    """
    Fully local RAG engine. No cloud calls, no API keys, no data leaves your machine.

    Architecture:
        User files → parse → chunk → embed (MiniLM) → store (ChromaDB)
        Query → embed → similarity search → top-K chunks → LLM answer (Ollama)
    """

    # -----------------------------------------------------------------------
    # CONFIGURATION — all tunables in one place
    # -----------------------------------------------------------------------
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 22MB, 384-dim, Apache 2.0
    LLM_MODEL = "qwen2:1.5b"               # 1.5B params, ~1GB RAM
    CHUNK_SIZE = 500                         # chars per chunk
    CHUNK_OVERLAP = 150                      # overlap between chunks (30%)
    BATCH_SIZE = 20                          # chunks per ChromaDB upsert
    MAX_FILE_SIZE_MB = 100                   # skip files larger than this
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    # Directories to skip during scanning — system/dev folders that waste time
    EXCLUDE_DIRS = frozenset({
        "AppData", "Local Settings", "Application Data",
        "Windows", "Program Files", "Program Files (x86)",
        "node_modules", ".git", ".vscode", ".idea",
        "venv", "env", "__pycache__", "site-packages",
        ".cache", ".npm", ".cargo", "dist", "build",
    })

    def __init__(self, db_path: str = "local_rag_db"):
        """
        Initialize the RAG engine.

        Args:
            db_path: Where ChromaDB stores its SQLite file on disk.
                     Relative to the working directory.

        WHY we load the embedding model here (not lazily):
            - Streamlit caches the RAG instance via @st.cache_resource,
              so this __init__ runs only once per app lifetime.
            - Loading eagerly means the first search isn't slow.
        """
        logger.info("Initializing LocalRAG engine...")

        # --- Embedding model ---
        # sentence-transformers auto-downloads on first run (~22MB),
        # then caches locally in ~/.cache/huggingface/
        logger.info(f"Loading embedding model: {self.EMBEDDING_MODEL}")
        self.encoder = SentenceTransformer(self.EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully.")

        # --- Vector database ---
        # PersistentClient writes to disk automatically. No manual save needed.
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="local_files",
            metadata={"hnsw:space": "cosine"},  # cosine similarity for search
        )
        logger.info(f"ChromaDB initialized at ./{db_path}")

    # -----------------------------------------------------------------------
    # INDEX MANAGEMENT
    # -----------------------------------------------------------------------

    def reset_index(self) -> None:
        """
        Wipe the entire vector index and recreate an empty collection.

        WHEN TO USE:
            - After switching embedding models (vectors are incompatible)
            - If the index gets corrupted (rare, but ChromaDB HNSW can break)
            - When you want a clean slate
        """
        logger.warning("Resetting index — all indexed data will be lost.")
        self.client.delete_collection("local_files")
        self.collection = self.client.get_or_create_collection(
            name="local_files",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Index reset complete.")

    def get_indexed_stats(self) -> Dict:
        """
        Return statistics about the current index.

        Used by the UI to show how many documents/chunks are indexed.
        """
        count = self.collection.count()
        return {
            "total_chunks": count,
            "embedding_model": self.EMBEDDING_MODEL,
            "llm_model": self.LLM_MODEL,
        }

    # -----------------------------------------------------------------------
    # FILE PARSING
    # -----------------------------------------------------------------------

    @staticmethod
    def get_file_hash(filepath: str) -> str:
        """
        Calculate MD5 hash of a file for deduplication.

        WHY MD5: Fast, and we only need change detection, not cryptographic
        security. SHA-256 would be 2x slower for zero benefit here.
        """
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                hash_md5.update(block)
        return hash_md5.hexdigest()

    def parse_file(self, filepath: Path) -> Generator[str, None, None]:
        """
        Extract text from a file and yield it as overlapping chunks.

        SUPPORTED FORMATS:
            - PDF  → PyMuPDF (fitz) extracts text page-by-page
            - DOCX → python-docx reads paragraphs
            - TXT  → plain read with UTF-8 fallback

        WHY GENERATOR (yield):
            Memory efficiency. A 50-page PDF might produce hundreds of chunks.
            Yielding one at a time means we never hold the entire document
            in memory — critical on 8GB RAM.
        """
        try:
            ext = filepath.suffix.lower()

            if ext == ".pdf":
                yield from self._parse_pdf(filepath)
            elif ext == ".docx":
                yield from self._parse_docx(filepath)
            elif ext == ".txt":
                yield from self._parse_txt(filepath)
            else:
                logger.warning(f"Unsupported file type: {ext} ({filepath.name})")

        except Exception as e:
            logger.error(f"Error parsing {filepath.name}: {e}")

    def _parse_pdf(self, filepath: Path) -> Generator[str, None, None]:
        """
        Parse PDF using PyMuPDF (fitz).

        WHY PyMuPDF over PyPDF2/pdfplumber:
            - 10x faster than PyPDF2 for text extraction
            - Lower memory usage than pdfplumber
            - Better handling of complex layouts
        """
        try:
            with fitz.open(filepath) as doc:
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():
                        yield from self._chunk_text(text)
        except Exception as e:
            logger.error(f"PDF parse error ({filepath.name}): {e}")

    def _parse_docx(self, filepath: Path) -> Generator[str, None, None]:
        """Parse DOCX using python-docx, extracting paragraph by paragraph."""
        try:
            doc = Document(filepath)
            # Accumulate paragraphs into a single text block, then chunk.
            # WHY: DOCX paragraphs can be very short (single lines), so
            # chunking paragraph-by-paragraph would create too-small chunks.
            full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            if full_text:
                yield from self._chunk_text(full_text)
        except Exception as e:
            logger.error(f"DOCX parse error ({filepath.name}): {e}")

    def _parse_txt(self, filepath: Path) -> Generator[str, None, None]:
        """Parse plain text files with encoding fallback."""
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            if text.strip():
                yield from self._chunk_text(text)
        except Exception as e:
            logger.error(f"TXT parse error ({filepath.name}): {e}")

    # -----------------------------------------------------------------------
    # TEXT CHUNKING
    # -----------------------------------------------------------------------

    def _chunk_text(self, text: str) -> Generator[str, None, None]:
        """
        Split text into overlapping chunks using a sliding window.

        PARAMETERS:
            chunk_size = 500 chars
            overlap    = 150 chars (30%)

        WHY SLIDING WINDOW:
            - Simpler and faster than sentence-level splitting (no spaCy/NLTK)
            - 150 char overlap ensures context isn't lost at boundaries
            - We try to break at word boundaries to avoid splitting mid-word

        WHY 500 CHARS (not 1000):
            - Smaller chunks = more precise search results
            - MiniLM-L6-v2 handles short texts well
            - Tradeoff: more chunks → slightly more storage and slower indexing
              (acceptable for ~55% accuracy target)

        EXAMPLE:
            Text: "The quick brown fox jumps over the lazy dog. Pack my box..."
            Chunk 1: "The quick brown fox jumps..."  (0–500)
            Chunk 2: "...fox jumps over the lazy..." (350–850) ← 150 char overlap
        """
        if not text or not text.strip():
            return

        # Normalize whitespace (collapse multiple spaces/newlines)
        text = " ".join(text.split())

        # Short text: return as single chunk
        if len(text) <= self.CHUNK_SIZE:
            yield text
            return

        start = 0
        while start < len(text):
            end = start + self.CHUNK_SIZE
            chunk = text[start:end]

            # Try to break at a word boundary (space) to avoid mid-word splits
            # Only look in the last 20% of the chunk for a space
            if end < len(text):
                last_space = chunk.rfind(" ")
                if last_space != -1 and last_space > self.CHUNK_SIZE * 0.8:
                    end = start + last_space
                    chunk = text[start:end]

            yield chunk.strip()

            # Advance by (chunk_length - overlap)
            # This creates the sliding window effect
            advance = len(chunk) - self.CHUNK_OVERLAP
            if advance <= 0:
                advance = 1  # Safety: always advance at least 1 char
            start += advance

    # -----------------------------------------------------------------------
    # INDEXING (Scan folders → Parse → Chunk → Embed → Store)
    # -----------------------------------------------------------------------

    def list_common_folders(self) -> List[Dict[str, str]]:
        """
        Return a list of common user folders that likely contain documents.

        Used by the UI to show pre-populated folder checkboxes.
        WHY: Users shouldn't have to remember folder paths. Desktop,
        Documents, Downloads are where most people store files.
        """
        home = Path.home()
        candidates = [
            {"name": "📄 Documents", "path": str(home / "Documents")},
            {"name": "📥 Downloads", "path": str(home / "Downloads")},
            {"name": "🖥️ Desktop",   "path": str(home / "Desktop")},
            {"name": "🖼️ Pictures",  "path": str(home / "Pictures")},
            {"name": "🏠 Home",      "path": str(home)},
        ]
        # Only return folders that actually exist on this machine
        return [f for f in candidates if os.path.isdir(f["path"])]

    def scan_and_index(
        self,
        folders: List[str],
        progress_callback: Optional[Callable] = None,
        stop_event=None,
    ) -> Dict[str, int]:
        """
        Scan multiple folders, parse files, and index them into ChromaDB.

        ALGORITHM:
            Phase 1: Walk all folders, collect valid file paths (pre-scan)
            Phase 2: For each file, hash → check if already indexed → parse → chunk → embed → store

        WHY TWO PHASES:
            Phase 1 gives us a total file count so we can show accurate
            progress percentages in the UI. Without it, we'd only know
            "processing file N" but not "of M total".

        Args:
            folders:           List of absolute folder paths to scan
            progress_callback: Function(dict) called with progress updates for the UI
            stop_event:        threading.Event to signal early stop (kill switch)

        Returns:
            Dict with indexing statistics (files_found, files_indexed, files_skipped, etc.)
        """
        stats = {
            "files_found": 0,
            "files_indexed": 0,
            "files_skipped": 0,
            "files_skipped_list": [],
            "files_failed": 0,
            "chunks_created": 0,
            "phase": "scanning",
        }

        def report(msg: str, **extra):
            """Send a progress update to the UI."""
            if progress_callback:
                progress_callback({**stats, "message": msg, **extra})

        # ---- PHASE 1: Pre-scan to collect all valid file paths ----
        report("Scanning folders for documents...")
        all_files: List[Path] = []

        for folder in folders:
            if stop_event and stop_event.is_set():
                break

            folder_path = Path(folder)
            if not folder_path.exists():
                logger.warning(f"Folder does not exist: {folder}")
                continue

            for root, dirs, files in os.walk(folder_path):
                if stop_event and stop_event.is_set():
                    break

                # Prune excluded directories IN-PLACE (os.walk respects this)
                dirs[:] = [
                    d for d in dirs
                    if d not in self.EXCLUDE_DIRS and not d.startswith(".")
                ]

                for fname in files:
                    if fname.startswith(".") or fname.startswith("~$"):
                        continue  # Skip hidden/temp files

                    fpath = Path(root) / fname
                    if fpath.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                        continue

                    try:
                        size_mb = fpath.stat().st_size / (1024 * 1024)
                        if size_mb > self.MAX_FILE_SIZE_MB:
                            reason = f"Too large ({size_mb:.1f}MB)"
                            logger.info(f"Skipping: {fpath.name} - {reason}")
                            stats["files_skipped"] += 1
                            stats["files_skipped_list"].append({"name": fpath.name, "reason": reason})
                            continue
                        all_files.append(fpath)
                    except OSError as e:
                        logger.error(f"Cannot access {fpath}: {e}")

        stats["files_found"] = len(all_files)
        stats["phase"] = "indexing"

        if not all_files:
            report("No supported files found in the selected folders.")
            return stats

        report(f"Found {len(all_files)} files. Starting indexing...")
        logger.info(f"Phase 2: Indexing {len(all_files)} files")

        # ---- PHASE 2: Index each file ----
        batch_docs: List[str] = []
        batch_metas: List[Dict] = []
        batch_ids: List[str] = []

        for file_idx, filepath in enumerate(all_files):
            if stop_event and stop_event.is_set():
                report("Indexing stopped by user.")
                break

            try:
                # Progress calculation
                percent = int(((file_idx + 1) / len(all_files)) * 100)
                report(
                    f"Processing: {filepath.name}",
                    current_file=filepath.name,
                    current_index=file_idx + 1,
                    total_files=len(all_files),
                    percent=percent,
                )

                # Check if file is already indexed (via content hash)
                file_hash = self.get_file_hash(str(filepath))
                check_id = f"{filepath.name}_{file_hash}_0"

                try:
                    existing = self.collection.get(ids=[check_id])
                    if existing and existing["ids"]:
                        stats["files_skipped"] += 1
                        logger.info(f"Skipping (already indexed): {filepath.name}")
                        continue
                except Exception:
                    pass  # If check fails, proceed with indexing

                # Parse file → chunks
                chunk_idx = 0
                for chunk in self.parse_file(filepath):
                    if stop_event and stop_event.is_set():
                        break

                    batch_docs.append(chunk)
                    batch_metas.append({
                        "source": str(filepath),
                        "filename": filepath.name,
                        "hash": file_hash,
                        "extension": filepath.suffix.lower(),
                    })
                    batch_ids.append(f"{filepath.name}_{file_hash}_{chunk_idx}")
                    chunk_idx += 1

                    # Process batch when full
                    if len(batch_docs) >= self.BATCH_SIZE:
                        self._process_batch(batch_docs, batch_metas, batch_ids)
                        stats["chunks_created"] += len(batch_docs)
                        batch_docs, batch_metas, batch_ids = [], [], []
                        # REAL-TIME UPDATE: report progress while processing chunks in the same file
                        report(
                            f"Chunking: {filepath.name} ({stats['chunks_created']} chunks total)",
                            current_file=filepath.name,
                            current_index=file_idx + 1,
                            total_files=len(all_files),
                            percent=percent,
                        )

                if chunk_idx > 0:
                    stats["files_indexed"] += 1
                    logger.info(f"Indexed {filepath.name}: {chunk_idx} chunks")

                # Periodic garbage collection to keep memory in check
                # WHY: On 8GB RAM, Python's GC may not run often enough.
                # Manual GC every 25 files prevents memory buildup.
                if (file_idx + 1) % 25 == 0:
                    gc.collect()

            except Exception as e:
                stats["files_failed"] += 1
                logger.error(f"Failed to process {filepath.name}: {e}")

        # Flush remaining batch
        if batch_docs and not (stop_event and stop_event.is_set()):
            self._process_batch(batch_docs, batch_metas, batch_ids)
            stats["chunks_created"] += len(batch_docs)

        stats["phase"] = "done"
        report(f"Done! Indexed {stats['files_indexed']} files ({stats['chunks_created']} chunks).")
        logger.info(f"Indexing complete: {stats}")
        return stats

    def _process_batch(
        self,
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str],
    ) -> None:
        """
        Encode a batch of text chunks and upsert them into ChromaDB.

        WHY BATCHING:
            - Encoding 20 chunks at once is faster than 20 individual calls
              (GPU/CPU SIMD parallelism in the encoder)
            - ChromaDB upsert is more efficient with batches
            - Batch size of 20 balances memory usage vs throughput

        WHY UPSERT (not insert):
            - If we re-index a file, existing chunks are updated in-place
              rather than creating duplicates.
        """
        if not documents:
            return

        try:
            embeddings = self.encoder.encode(documents).tolist()
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as e:
            logger.error(f"Batch upsert error: {e}")
            # Check for index corruption (HNSW errors)
            err_str = str(e).lower()
            if "hnsw" in err_str or "loading index" in err_str:
                logger.critical("HNSW index corruption detected. Resetting index.")
                self.reset_index()
                # Retry once
                try:
                    embeddings = self.encoder.encode(documents).tolist()
                    self.collection.upsert(
                        ids=ids,
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas,
                    )
                except Exception as retry_err:
                    logger.error(f"Retry after reset also failed: {retry_err}")

    # -----------------------------------------------------------------------
    # SEARCH
    # -----------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.45,
    ) -> Dict:
        """
        Search indexed documents for chunks similar to the query.

        HOW IT WORKS:
            1. Encode the query text into a 384-dim vector (same model as indexing)
            2. ChromaDB finds the k*2 nearest vectors using HNSW algorithm
            3. Filter results by cosine distance threshold
            4. Return top-k results with metadata

        THRESHOLD EXPLAINED (cosine distance):
            - 0.0 = identical match
            - 1.0 = orthogonal (unrelated)
            - 2.0 = opposite meaning
            Default 0.45 aims for ~55% precision: accepts moderately similar
            results without being too loose. Lower = stricter, higher = more results.

        WHY k*2 THEN FILTER:
            Fetching extra results before threshold filtering ensures we still
            get k results even after filtering out low-quality matches.

        Args:
            query:           Natural language search string
            k:               Max number of results to return
            score_threshold: Max cosine distance (lower = stricter)

        Returns:
            Dict with keys: ids, documents, metadatas, distances
        """
        if self.collection.count() == 0:
            return self._empty_results()

        query_embedding = self.encoder.encode([query]).tolist()

        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(k * 2, self.collection.count()),
            )
        except Exception as e:
            logger.error(f"Search error: {e}")
            return self._empty_results()

        # Filter by distance threshold
        filtered = self._empty_results()

        if results["distances"] and results["distances"][0]:
            for i, dist in enumerate(results["distances"][0]):
                if dist <= score_threshold:
                    filtered["ids"][0].append(results["ids"][0][i])
                    filtered["documents"][0].append(results["documents"][0][i])
                    filtered["metadatas"][0].append(results["metadatas"][0][i])
                    filtered["distances"][0].append(dist)

            # Limit to k results
            for key in filtered:
                if filtered[key] and filtered[key][0]:
                    filtered[key][0] = filtered[key][0][:k]

        return filtered

    @staticmethod
    def _empty_results() -> Dict:
        """Return an empty results structure."""
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    # -----------------------------------------------------------------------
    # ANSWER GENERATION (Local LLM via Ollama)
    # -----------------------------------------------------------------------

    def generate_answer(self, query: str, context_docs: List[str]) -> str:
        """
        Generate a natural language answer using the local LLM (Ollama).

        HOW IT WORKS:
            1. Concatenate the top retrieved document chunks as "context"
            2. Construct a prompt: "Given this context, answer this question"
            3. Send to Ollama's local HTTP API (127.0.0.1:11434)
            4. Return the generated text

        WHY OLLAMA:
            - Runs LLMs locally with zero cloud dependency
            - Simple REST API, no complex setup
            - Handles model quantization automatically (q4 for low RAM)

        WHY qwen2:1.5b:
            - ~1GB RAM usage with q4 quantization
            - Reasonable instruction-following for a 1.5B model
            - Starting small — can upgrade to phi3:mini (3.8B) if RAM allows

        Args:
            query:        The user's question
            context_docs: Retrieved document chunks to ground the answer

        Returns:
            Generated answer string, or error message
        """
        if not context_docs:
            return "No relevant documents found to answer your question."

        # Limit context to prevent exceeding the model's context window
        # qwen2:1.5b has a ~4K token context window
        context_text = "\n\n---\n\n".join(context_docs[:3])

        prompt = (
            "You are a helpful and conversational AI assistant. Answer the user's question using the information from the Context provided.\n"
            "DO NOT just return a title or a few words. Provide a full, natural response in complete sentences.\n"
            "If the context contains the answer, explain it clearly.\n"
            "If the context is unrelated, simply say that the provided documents don't have that information.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Conversational Answer:"
        )

        try:
            response = ollama.chat(
                model=self.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]
        except Exception as e:
            error_msg = str(e)
            if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                return (
                    "⚠️ Cannot connect to Ollama. Make sure Ollama is running:\n"
                    "1. Open a terminal\n"
                    "2. Run: ollama serve\n"
                    "3. In another terminal: ollama pull qwen2:1.5b"
                )
            return f"Error generating answer: {error_msg}"


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rag = LocalRAG()
    print("RAG engine initialized successfully.")
    print(f"Index stats: {rag.get_indexed_stats()}")
    print(f"Common folders: {[f['name'] for f in rag.list_common_folders()]}")
