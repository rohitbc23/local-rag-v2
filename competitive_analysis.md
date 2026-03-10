# Competitive Analysis — Local RAG v2 vs Established Projects

## How to Read This Document

This analysis compares **Local RAG v2** (what we built) against 8 established open-source projects that solve similar problems. The goal is to help you understand where we stand, what we do better, and what we sacrifice.

> **TL;DR**: We built a lightweight, single-purpose tool optimized for your 8GB hardware. The competitors are more feature-rich but demand 2-4x more RAM and significantly more setup complexity. Our project is the equivalent of a sharp pocketknife — the others are Swiss Army knives.

---

## Head-to-Head Comparison

| Feature | **Local RAG v2** (Ours) | **PrivateGPT** | **AnythingLLM** | **LocalGPT** | **Khoj** | **Danswer (Onyx)** | **Quivr** | **RAGFlow** |
|---|---|---|---|---|---|---|---|---|
| **Min RAM** | **~1.3GB** | ~10GB+ | ~2GB (wrapper only) | 16GB+ | 8GB+ | 16GB+ | 8GB+ | 16GB+ |
| **GPU Required** | ❌ No | Recommended | Optional | Recommended | Optional | Optional | Optional | Yes (for OCR) |
| **Setup Steps** | **3** (pip, ollama pull, run) | 10+ (Poetry, cmake, config) | 5-8 (Docker or installer) | 8+ (Docker, GPU drivers) | 6+ (Docker compose) | 10+ (Docker, Postgres, Redis) | 8+ (Docker, Supabase) | 10+ (Docker, MinIO, MySQL) |
| **Runs on 8GB RAM** | ✅ **Yes** | ⚠️ Barely | ✅ (if using remote LLM) | ❌ No | ⚠️ Barely | ❌ No | ⚠️ Barely | ❌ No |
| **Fully Offline** | ✅ Yes | ✅ Yes | ⚠️ Configurable | ✅ Yes | ⚠️ Configurable | ⚠️ Configurable | ⚠️ Configurable | ⚠️ Configurable |
| **File Types** | PDF, DOCX, TXT | PDF, DOCX, TXT + more | 20+ formats | PDF (mainly) | PDF, MD, TXT, Org | 30+ formats | 15+ formats | PDF, DOCX, XLSX, PPTX |
| **Vector DB** | ChromaDB (embedded) | Qdrant/Chroma/PGVector | LanceDB/Chroma/Pinecone | ChromaDB | PostgreSQL (pgvector) | Vespa | PGVector | Elasticsearch |
| **LLM Integration** | Ollama (qwen2:1.5b) | Ollama/LlamaCPP/OpenAI | Ollama/LMStudio/OpenAI | Ollama/LlamaCPP | Ollama/OpenAI/Groq | Any LLM (API-based) | Ollama/OpenAI | Ollama/OpenAI/Custom |
| **UI** | Streamlit (web) | Gradio (web) | Custom desktop/web app | Streamlit (web) | React web app | React web app | React web app | React web app |
| **Folder Selection** | ✅ Checkbox picker | ❌ Upload only | ✅ Upload/drag-drop | ❌ CLI config | ❌ Upload + integrations | ✅ Connectors | ✅ Upload | ✅ Upload |
| **Progress Tracking** | ✅ Real-time % bar | ❌ Minimal | ⚠️ Basic | ❌ CLI logs | ❌ Background | ⚠️ Basic | ⚠️ Basic | ✅ Yes |
| **Hybrid Search** | ❌ Semantic only | ❌ Semantic only | ❌ Semantic only | ✅ Dense + keyword | ❌ Semantic only | ✅ Semantic + keyword | ⚠️ Basic | ✅ Semantic + keyword |
| **Chat Memory** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Multi-user** | ❌ No | ❌ No | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes (RBAC) | ✅ Yes | ✅ Yes |
| **API Available** | ❌ No | ✅ FastAPI | ✅ REST API | ❌ No | ✅ REST API | ✅ REST API | ✅ REST API | ✅ REST API |
| **License** | All Apache/MIT | Apache 2.0 | MIT | Apache 2.0 | AGPL-3.0 | MIT/AGPL | Apache 2.0 | Apache 2.0 |
| **GitHub Stars** | N/A (custom) | ~57K | ~56K | ~22K | ~33K | ~17K | ~39K | ~74K |
| **Codebase Size** | **~600 lines** | ~15K lines | ~50K+ lines | ~5K lines | ~20K lines | ~100K+ lines | ~30K+ lines | ~50K+ lines |

---

## Detailed Competitor Breakdown

### 1. PrivateGPT
**What it is:** A production-grade RAG framework by Zylon, built on LlamaIndex + FastAPI.

| Pros | Cons |
|---|---|
| ✅ API-first — great for building on top of | ❌ Needs ~10GB+ free RAM minimum |
| ✅ Highly modular (swap any component) | ❌ Complex setup (Poetry, cmake, C++ compiler) |
| ✅ Best documentation of all competitors | ❌ Gradio UI is functional but not polished |
| ✅ Enterprise-ready architecture | ❌ Overkill for personal document search |
| ✅ Supports multiple vector stores | ❌ Steeper learning curve |

**Best for:** Developers building a custom product on top of a RAG framework.

---

### 2. AnythingLLM
**What it is:** An all-in-one desktop app that bundles RAG, agents, and chat into one installer.

| Pros | Cons |
|---|---|
| ✅ Polished desktop app (Electron) | ❌ Electron = 200MB+ baseline RAM |
| ✅ Easiest setup of all competitors | ❌ "Wrapper" design means LLM runs elsewhere |
| ✅ Supports 20+ file formats | ❌ Not truly self-contained unless configured |
| ✅ Agent capabilities (web scraping, etc.) | ❌ Can be confusing which parts are local |
| ✅ Multi-user workspaces | ❌ Limited advanced RAG features |

**Best for:** Non-technical users who want a "just works" chat-with-documents app.

---

### 3. LocalGPT
**What it is:** A focused local RAG tool with advanced retrieval features (hybrid search, smart routing).

| Pros | Cons |
|---|---|
| ✅ Hybrid search (semantic + keyword) | ❌ Requires 16GB+ RAM minimum |
| ✅ Smart router (decides RAG vs direct LLM) | ❌ GPU strongly recommended |
| ✅ Answer verification and context pruning | ❌ Primarily supports PDF only |
| ✅ Two GUI options | ❌ Docker-based setup |
| ✅ Closest to our project in scope | ❌ Occasional formatting bugs reported |

**Best for:** Users with powerful hardware who want the most accurate local RAG retrieval.

---

### 4. Khoj
**What it is:** An "AI second brain" with deep integrations into note-taking tools (Obsidian, Emacs).

| Pros | Cons |
|---|---|
| ✅ Obsidian/Emacs plugins — seamless workflow | ❌ Docker Compose setup |
| ✅ Web search integration alongside RAG | ❌ Primarily designed for notes/knowledge base |
| ✅ WhatsApp and Telegram integrations | ❌ Less focused on document search |
| ✅ Self-hosting with user management | ❌ AGPL license (restrictive for commercial use) |
| ✅ Conversation memory | ❌ Requires PostgreSQL |

**Best for:** Knowledge workers who live in Obsidian/Emacs and want AI across their notes.

---

### 5. Danswer (now Onyx)
**What it is:** An enterprise-grade AI assistant with 30+ data source connectors (Slack, Confluence, Google Drive, etc.).

| Pros | Cons |
|---|---|
| ✅ 30+ connectors (Slack, Drive, SharePoint) | ❌ 16GB+ RAM, Docker, Postgres, Redis |
| ✅ Hybrid search out-of-the-box | ❌ Enterprise complexity |
| ✅ Role-based access control (RBAC) | ❌ Not designed for local file search |
| ✅ Multi-pass indexing for accuracy | ❌ Heavy infrastructure (Vespa, Celery) |
| ✅ Production-proven at companies | ❌ Way too heavy for 8GB personal use |

**Best for:** Teams/companies needing AI search across their SaaS tools.

---

### 6. Quivr
**What it is:** An open-source "second brain" platform with customizable AI assistants ("Brains").

| Pros | Cons |
|---|---|
| ✅ Custom "Brains" — train AI on specific topics | ❌ Requires Supabase + Docker |
| ✅ Clean, modern React UI | ❌ Complex self-hosting |
| ✅ 15+ file format support | ❌ 8GB+ RAM needed |
| ✅ API access for integrations | ❌ Cloud-first design (self-hosting is secondary) |
| ✅ Active community (~39K stars) | ❌ Frequent breaking changes reported |

**Best for:** Users wanting a personal ChatGPT-like experience with their own documents.

---

### 7. RAGFlow
**What it is:** A RAG engine focused on deep document understanding — tables, layouts, diagrams.

| Pros | Cons |
|---|---|
| ✅ Best document parsing (tables, layouts) | ❌ 16GB+ RAM, GPU recommended |
| ✅ Handles complex PDFs (invoices, forms) | ❌ Heavy Docker stack (MinIO, MySQL, Redis) |
| ✅ Hybrid search (semantic + keyword) | ❌ Complex setup |
| ✅ Multi-modal (text + visual elements) | ❌ Overkill for simple text search |
| ✅ Production-ready at scale | ❌ Not designed for lightweight personal use |

**Best for:** Users with complex documents (financial reports, forms with tables) and powerful hardware.

---

## Where Local RAG v2 Wins

| Advantage | Why It Matters For You |
|---|---|
| **🪶 Lightest footprint** (~1.3GB RAM) | Actually runs on your 8GB machine without crashing |
| **⚡ Simplest setup** (3 commands) | No Docker, no databases, no config files |
| **📁 Folder-based indexing** with checkbox picker | You choose exactly which folders to scan |
| **📊 Real-time progress bar** | You can see exactly what's happening during indexing |
| **🔍 Every line of code is documented** | You understand and control every decision |
| **🔒 Zero cloud dependencies** | No Traceloop, no API tokens, no data leaks |
| **💰 Zero cost** | No API credits, no subscriptions |

---

## Where Local RAG v2 Loses

| Limitation | Which Competitor Does It Better |
|---|---|
| **No hybrid search** (semantic only) | LocalGPT, RAGFlow, Danswer have semantic + keyword |
| **No chat memory** (each query is independent) | All competitors support conversation history |
| **Limited file types** (PDF, DOCX, TXT) | AnythingLLM (20+), Danswer (30+), RAGFlow (tables) |
| **No API** (UI-only access) | PrivateGPT, AnythingLLM, Khoj expose REST APIs |
| **Single user** | AnythingLLM, Khoj, Danswer support multi-user |
| **Smaller LLM** (qwen2:1.5b) | Others can run 7B-70B models with more RAM |
| **No integrations** | Khoj has Obsidian/Emacs; Danswer has Slack/Drive |
| **No OCR** for scanned PDFs | RAGFlow handles visual/table extraction |

---

## Recommendation Matrix

| If you want... | Use... |
|---|---|
| 🪶 Lightweight local search on 8GB RAM | **Local RAG v2** (ours) |
| 🏢 Enterprise search across SaaS tools | **Danswer (Onyx)** |
| 🧠 AI copilot for Obsidian notes | **Khoj** |
| 📦 Easy desktop app, no coding | **AnythingLLM** |
| 🔧 Framework to build your own product | **PrivateGPT** |
| 📊 Complex documents (tables, invoices) | **RAGFlow** |
| 🎯 Best retrieval accuracy (with 16GB+) | **LocalGPT** |
| 🧩 Custom AI "Brains" per topic | **Quivr** |

---

## Potential Upgrades for Local RAG v2

Based on this analysis, here are features worth adding (in priority order):

1. **Chat memory** — Store conversation history for follow-up questions
2. **Hybrid search** — Add BM25 keyword search alongside semantic (ChromaDB supports this)
3. **More file types** — Add Markdown, CSV, PPTX support
4. **Upgrade LLM** — Move to `phi3:mini` (3.8B) if RAM allows
5. **REST API** — Add a FastAPI layer for programmatic access
