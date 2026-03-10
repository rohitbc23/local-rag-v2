"""
app.py — Streamlit UI for Local RAG Search
===========================================

UI/UX DESIGN DECISIONS:
    1. SIDEBAR: Contains indexing controls and settings — keeps the main area
       clean for search results.
    2. FOLDER PICKER: Checkboxes for common folders (Documents, Downloads, etc.)
       + a text input for custom paths. Users shouldn't have to remember paths.
    3. PROGRESS BAR: Shows real % completion with file count and current file name.
       Uses Streamlit's native progress bar with text overlay.
    4. KILL SWITCH: Stop button during indexing (threading.Event-based).
    5. INDEX STATS: Shows chunk count and model info so user knows what's indexed.
    6. SEARCH RESULTS: Confidence color-coding (green/orange/red), snippet
       highlighting, "Open File" buttons.
    7. AI ANSWER: LLM-generated answer displayed prominently above search results.
    8. CUSTOM CSS: Dark-themed, premium look with subtle gradients and shadows.
"""

import streamlit as st
import os
import re
import time
import threading

os.environ["TRACELOOP_TELEMETRY"] = "false"

from traceloop.sdk import Traceloop
from dotenv import load_dotenv

load_dotenv()

dt_token = os.environ.get("DYNATRACE_API_TOKEN", "")
dt_url = os.environ.get("DYNATRACE_TENANT_URL", "")
headers = { "Authorization": f"Api-Token {dt_token}" }
Traceloop.init(
    app_name="on-device-RAG",
    api_endpoint=dt_url,
    headers=headers
)
# ---------------------------------------------------------------------------
# PAGE CONFIG — Must be the first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Local RAG Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CUSTOM CSS — Premium dark theme with modern aesthetics
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
        padding-bottom: 0;
    }

    .sub-header {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-top: 0;
        padding-top: 0;
    }

    /* Result card styling */
    .result-card {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
        border: 1px solid #3a3a5a;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }

    /* Confidence badge */
    .confidence-high {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .confidence-mid {
        background: linear-gradient(135deg, #ed8936, #dd6b20);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .confidence-low {
        background: linear-gradient(135deg, #fc8181, #e53e3e);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* AI answer box */
    .ai-answer-box {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid #667eea;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    /* Stats pill */
    .stats-pill {
        background: #2a2a3e;
        border: 1px solid #3a3a5a;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        text-align: center;
        margin: 0.25rem 0;
    }

    /* File type badge */
    .file-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .file-badge-pdf { background: #e53e3e22; color: #fc8181; border: 1px solid #e53e3e44; }
    .file-badge-docx { background: #3182ce22; color: #63b3ed; border: 1px solid #3182ce44; }
    .file-badge-txt { background: #38a16922; color: #68d391; border: 1px solid #38a16944; }

    /* Progress area */
    .indexing-status {
        background: #1a1a2e;
        border: 1px solid #3a3a5a;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }

    /* Search input styling */
    .stTextInput>div>div>input {
        background-color: #1e1e2e;
        border: 2px solid #3a3a5a;
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 1.05rem;
        transition: border-color 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# INITIALIZE RAG ENGINE (cached — runs once per app lifetime)
# ---------------------------------------------------------------------------
from rag_engine import LocalRAG

@st.cache_resource
def get_rag_engine():
    """
    WHY @st.cache_resource:
        The RAG engine loads an embedding model (~80MB RAM) and opens a
        ChromaDB connection. We only want to do this ONCE, not on every
        Streamlit rerun (which happens on any UI interaction).
    """
    return LocalRAG()

rag = get_rag_engine()


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown('<h1 class="main-header">Local Device Search 🔍</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Search your local documents with AI — fully private, fully offline.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------------------------
# WHY session state: Streamlit reruns the entire script on every interaction.
# Session state persists values across reruns (like React state).
defaults = {
    "indexing_thread": None,
    "stop_event": threading.Event(),
    "indexing_progress": {},
    "indexing_done": False,
    "selected_folders": [],
    "custom_folders": [],
}
for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# SIDEBAR — Indexing Controls + Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📁 Folder Selection")
    st.caption("Select which folders to index for searching.")

    # ---- Common folder checkboxes ----
    common_folders = rag.list_common_folders()
    selected = []

    for folder_info in common_folders:
        checked = st.checkbox(
            folder_info["name"],
            key=f"folder_{folder_info['path']}",
            help=folder_info["path"],
        )
        if checked:
            selected.append(folder_info["path"])

    # ---- Custom folder input ----
    st.markdown("---")
    st.caption("Or pick a custom folder:")

    if st.button("📁 Select Folder", use_container_width=True):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        folder = filedialog.askdirectory(master=root)
        root.destroy()
        if folder and folder not in st.session_state["custom_folders"]:
            st.session_state["custom_folders"].append(folder)
            st.rerun()

    for i, folder in enumerate(list(st.session_state["custom_folders"])):
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.markdown(f"`{folder}`")
        with col2:
            if st.button("❌", key=f"remove_custom_{i}"):
                st.session_state["custom_folders"].pop(i)
                st.rerun()
        
        if os.path.isdir(folder):
            selected.append(folder)
        else:
            st.caption(f"⚠️ Not found: `{folder}`")

    st.session_state["selected_folders"] = selected

    # Show selected count
    if selected:
        st.success(f"✅ {len(selected)} folder(s) selected")
    else:
        st.info("No folders selected yet.")

    # ---- Indexing Controls ----
    st.markdown("---")
    st.markdown("## ⚡ Indexing")

    is_indexing = (
        st.session_state["indexing_thread"] is not None
        and st.session_state["indexing_thread"].is_alive()
    )

    # --- Thread function for background indexing ---
    def _run_indexing_thread(folders, stop_event):
        """
        Runs in a background thread so the UI stays responsive.

        WHY THREADING:
            Indexing can take minutes for large folders. If we ran it in the
            main thread, the entire Streamlit UI would freeze. Threading lets
            the UI keep updating (progress bar, stop button).

        The progress_callback writes to session_state, which the main thread
        reads on each rerun to update the progress bar.
        """
        def on_progress(progress_dict):
            st.session_state["indexing_progress"] = progress_dict

        try:
            result = rag.scan_and_index(
                folders=folders,
                progress_callback=on_progress,
                stop_event=stop_event,
            )
            st.session_state["indexing_progress"] = {
                **result,
                "message": f"✅ Done! Indexed {result['files_indexed']} files "
                           f"({result['chunks_created']} chunks). "
                           f"Skipped {result['files_skipped']} already-indexed files.",
                "percent": 100,
                "phase": "done",
            }
        except Exception as e:
            st.session_state["indexing_progress"] = {
                "message": f"❌ Error: {e}",
                "phase": "error",
                "percent": 0,
            }
        finally:
            st.session_state["indexing_done"] = True

    # Indexing buttons
    col_start, col_stop = st.columns(2)

    with col_start:
        if not is_indexing:
            if st.button("▶️ Start Indexing", use_container_width=True, type="primary"):
                if not selected:
                    st.error("Please select at least one folder.")
                else:
                    st.session_state["stop_event"].clear()
                    st.session_state["indexing_done"] = False
                    st.session_state["indexing_progress"] = {
                        "message": "Starting...",
                        "percent": 0,
                        "phase": "starting",
                    }
                    from streamlit.runtime.scriptrunner import add_script_run_ctx
                    t = threading.Thread(
                        target=_run_indexing_thread,
                        args=(selected, st.session_state["stop_event"]),
                        daemon=True,
                    )
                    add_script_run_ctx(t)
                    t.start()
                    st.session_state["indexing_thread"] = t
                    st.rerun()

    with col_stop:
        if is_indexing:
            if st.button("⏹️ Stop", use_container_width=True, type="secondary"):
                st.session_state["stop_event"].set()
                st.session_state["indexing_progress"]["message"] = "Stopping..."
        else:
            if st.button("🗑️ Reset Index", use_container_width=True):
                rag.reset_index()
                st.session_state["indexing_progress"] = {
                    "message": "🗑️ Index cleared!",
                    "percent": 0,
                    "phase": "reset",
                }
                st.rerun()

    # ---- Progress Display ----
    progress = st.session_state.get("indexing_progress", {})

    if progress:
        st.markdown("---")
        phase = progress.get("phase", "")
        msg = progress.get("message", "")
        percent = progress.get("percent", 0)

        # Progress bar
        if is_indexing or phase in ("scanning", "indexing"):
            st.progress(
                min(percent / 100.0, 1.0),
                text=f"{percent}% complete",
            )

            # Detailed status
            current_file = progress.get("current_file", "")
            current_idx = progress.get("current_index", 0)
            total_files = progress.get("total_files", 0)

            if current_file:
                st.markdown(
                    f'<div class="indexing-status">'
                    f'📄 <strong>{current_file}</strong><br/>'
                    f'File {current_idx} of {total_files}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Live stats during indexing
            files_indexed = progress.get("files_indexed", 0)
            files_skipped = progress.get("files_skipped", 0)
            chunks = progress.get("chunks_created", 0)

            stat_cols = st.columns(3)
            with stat_cols[0]:
                st.metric("Indexed", files_indexed)
            with stat_cols[1]:
                st.metric("Skipped", files_skipped)
            with stat_cols[2]:
                st.metric("Chunks", chunks)

        # Status message
        if msg:
            if "Error" in msg or "❌" in msg:
                st.error(msg)
            elif "Done" in msg or "✅" in msg:
                st.success(msg)
            elif "cleared" in msg or "🗑️" in msg:
                st.info(msg)
            elif "Stopping" in msg:
                st.warning(msg)
            else:
                st.info(msg)

        # Show skipped files details if any
        skipped_list = progress.get("files_skipped_list", [])
        if skipped_list and phase == "done":
            with st.expander(f"⚠️ {len(skipped_list)} Files Skipped", expanded=False):
                for item in skipped_list:
                    st.caption(f"• **{item['name']}**: {item['reason']}")

    # Auto-rerun while indexing is active (poll every 1 second)
    if is_indexing:
        time.sleep(1)
        st.rerun()
    elif st.session_state["indexing_thread"] is not None and st.session_state["indexing_done"]:
        # Thread finished — clean up
        st.session_state["indexing_thread"] = None
        st.session_state["indexing_done"] = False
        st.rerun()

    # ---- Index Stats ----
    st.markdown("---")
    st.markdown("## 📊 Index Status")
    stats = rag.get_indexed_stats()

    st.markdown(
        f'<div class="stats-pill">📦 <strong>{stats["total_chunks"]}</strong> chunks indexed</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="stats-pill">🧠 Model: <strong>{stats["embedding_model"]}</strong></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="stats-pill">🤖 LLM: <strong>{stats["llm_model"]}</strong></div>',
        unsafe_allow_html=True,
    )

    # ---- Search Settings ----
    st.markdown("---")
    st.markdown("## ⚙️ Search Settings")

    threshold = st.slider(
        "Relevance Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help=(
            "Controls how strict search matching is.\n\n"
            "**Lower** = Stricter (only very similar results)\n\n"
            "**Higher** = More relaxed (more results, possibly less relevant)\n\n"
            "Default 0.45 aims for ~55% precision."
        ),
    )

    num_results = st.slider(
        "Max Results",
        min_value=1,
        max_value=20,
        value=5,
        help="Maximum number of search results to display.",
    )


# ---------------------------------------------------------------------------
# MAIN AREA — Search + Results
# ---------------------------------------------------------------------------
st.markdown("---")

# Search input
query = st.text_input(
    "🔍 What are you looking for?",
    placeholder="e.g., 'invoice from October', 'project proposal 2024', 'salary details'...",
    label_visibility="visible",
)

if query:
    # ---- Execute search ----
    with st.spinner("🔍 Searching your documents..."):
        results = rag.search(query, k=num_results, score_threshold=threshold)

    has_results = results["documents"] and results["documents"][0]

    # ---- AI Answer Generation ----
    if has_results:
        with st.spinner("🤖 Generating AI answer..."):
            context_docs = results["documents"][0][:3]
            answer = rag.generate_answer(query, context_docs)

        st.markdown(
            f'<div class="ai-answer-box">'
            f'<h3 style="margin-top:0;">🤖 AI Answer</h3>'
            f'<p style="font-size: 1.05rem; line-height: 1.7;">{answer}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            "No matches found. Try:\n"
            "- Indexing your documents first (use the sidebar)\n"
            "- Increasing the relevance threshold\n"
            "- Using different search terms"
        )

    # ---- Search Results ----
    if has_results:
        st.markdown(f"### 📄 Source Documents ({len(results['documents'][0])} results)")

        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            full_path = meta["source"]
            filename = meta["filename"]
            extension = meta.get("extension", "").lstrip(".")

            # Calculate confidence: cosine distance 0 → 100%, 1.0 → 0%
            confidence = max(0, (1 - distance)) * 100

            # Confidence styling
            if confidence >= 60:
                badge_class = "confidence-high"
                conf_emoji = "🟢"
            elif confidence >= 40:
                badge_class = "confidence-mid"
                conf_emoji = "🟡"
            else:
                badge_class = "confidence-low"
                conf_emoji = "🔴"

            # File type badge
            badge_type = f"file-badge-{extension}" if extension in ("pdf", "docx", "txt") else ""

            # Smart snippet: try to find query terms and show surrounding context
            snippet = doc
            query_terms = query.lower().split()
            lower_doc = doc.lower()
            best_idx = -1

            for term in query_terms:
                idx = lower_doc.find(term)
                if idx != -1:
                    best_idx = idx
                    break

            if best_idx != -1:
                start = max(0, best_idx - 80)
                end = min(len(doc), start + 350)
                snippet = ("..." if start > 0 else "") + doc[start:end] + ("..." if end < len(doc) else "")
            else:
                snippet = doc[:350] + ("..." if len(doc) > 350 else "")

            # Highlight matching terms in snippet
            for term in query_terms:
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                snippet = pattern.sub(lambda m: f"**{m.group(0)}**", snippet)

            # ---- Render result card ----
            with st.container():
                st.markdown(
                    f'<div class="result-card">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">'
                    f'<span style="font-size:1.15rem; font-weight:600;">📄 {filename}</span>'
                    f'<span class="{badge_class}">{conf_emoji} {confidence:.0f}%</span>'
                    f'</div>'
                    f'<span class="file-badge {badge_type}">{extension.upper()}</span>'
                    f'<span style="color:#718096; margin-left:0.5rem; font-size:0.85rem;">{full_path}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Expandable snippet
                with st.expander(f"Preview snippet", expanded=(i < 2)):
                    st.markdown(snippet)

                # Open file button
                col_open, col_spacer = st.columns([1, 5])
                with col_open:
                    if st.button("📂 Open File", key=f"open_{i}"):
                        try:
                            os.startfile(full_path)
                        except Exception as e:
                            st.error(f"Could not open: {e}")

elif not query:
    # ---- Empty state ----
    stats = rag.get_indexed_stats()

    if stats["total_chunks"] == 0:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                "### 1️⃣ Select Folders\n"
                "Use the sidebar to pick which folders contain your documents."
            )
        with col2:
            st.markdown(
                "### 2️⃣ Index Files\n"
                "Click **Start Indexing** to scan and process your files."
            )
        with col3:
            st.markdown(
                "### 3️⃣ Search\n"
                "Type a natural language query to find relevant documents."
            )
    else:
        st.markdown("---")
        st.info(
            f"📦 **{stats['total_chunks']} chunks** indexed and ready to search. "
            "Type a query above to get started!"
        )


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "🔒 **Fully Local** — No data leaves your machine. "
    "Powered by MiniLM-L6-v2 (embeddings) + qwen2:1.5b (answers) + ChromaDB (storage). "
    "All open-source."
)
