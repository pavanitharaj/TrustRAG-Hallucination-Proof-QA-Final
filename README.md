# TrustRAG: Hallucination-Proof Document Q&A Bot 🛡️

TrustRAG is a trustworthy, enterprise-grade Retrieval-Augmented Generation (RAG) assistant designed for zero hallucinations. It answers user queries **exclusively** from uploaded PDF documents. If a question cannot be proven by the retrieved evidence, it triggers a strict guardrail:

> **"I don't know based on the provided documents."**

---

## 🏛️ System Architecture

```
[User PDF Uploads]
       │
       ▼
[Page-Aware Text Extraction]  ──  pypdf extracts text while preserving 1-indexed page numbers
       │
       ▼
[Semantic Chunking]           ──  600-char chunks with 120-char overlap, respecting sentence boundaries
       │
       ▼
[Local Embeddings]            ──  SentenceTransformers (all-MiniLM-L6-v2) 384-dimensional vectors
       │
       ▼
[Vector Database]             ──  FAISS IndexFlatIP (Cosine Similarity via L2 normalization)
       │
       ▼
[Top-k Retrieval]             ──  Retrieves top-4 relevant chunks with exact similarity scores
       │
       ▼
[Two-Stage Safety Verification]
 ├── Stage 1: Evidence Verification Auditor (Gemini 2.5 Flash @ temp=0.0)
 │     └── Evaluates: Does context contain direct, complete factual proof?
 │
 ├── IF UNSUPPORTED (Guardrail Triggered):
 │     └── Returns: "I don't know based on the provided documents."
 │         (No guessing, no fabricated speculation, citations suppressed)
 │
 └── IF SUPPORTED (Grounded Synthesis):
       ├── Gemini generates answer using ONLY verified facts (prohibited from making citations)
       └── Python programmatically generates citations:
           [Document Name | Page Number | Chunk ID | Relevance Score]
```

---

## 🔑 Key Engineering & Design Decisions

### 1. Deterministic Python Citations (Anti-Citation Hallucination)
LLMs frequently hallucinate source titles, page numbers, and chunk references when asked to generate their own footnotes. In TrustRAG, **the LLM is strictly prohibited from generating citations**. Citations are stamped deterministically by Python code using immutable vector metadata:
- `Document Name`
- `Page Number` (1-indexed directly from PDF reader)
- `Chunk ID`
- `Relevance Score` (cosine similarity percentage)

### 2. Two-Stage Independent Audit Before Generation
Instead of a single prompt trying to answer and verify simultaneously, TrustRAG separates the reasoning:
- **Verification Stage (Auditor)**: Operates at temperature `0.0` with JSON schema enforcement. It strictly determines if the chunks contain the answer. If false, answer generation is aborted immediately.
- **Generation Stage (Synthesizer)**: Answers strictly using the verified chunks with zero external extrapolation.

### 3. Sensible Chunk Size & Overlap
- **Chunk Size**: `600 characters` (~100-120 words). Small enough to maintain high semantic precision in vector search, large enough to contain complete factual sentences.
- **Chunk Overlap**: `120 characters` (~20 words). Prevents information fragmentation across boundaries.
- **Strict Page Boundary**: Chunks do not span across page boundaries, guaranteeing that every single chunk has an exact, unambiguous page citation.

### 4. Fully Free / Open-Source Local Vector Stack
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (runs locally, free, fast CPU inference).
- Vector Store: `FAISS` (IndexFlatIP with normalized vectors for exact cosine similarity). Includes a pure NumPy dot-product fallback if FAISS is not present.

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Virtual Environment

```bash
git clone <repo-url>
cd trustrag

python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY="your_actual_google_gemini_api_key_here"
```

*(Note: Never commit your `.env` file. It is excluded in `.gitignore`.)*

### 4. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

Streamlit will launch locally in your browser at `http://localhost:8501`.

---

## 🧪 Testing Rubric Scenarios

| Test Category | Example Query | Expected Behavior |
|---|---|---|
| **Answerable (Policy)** | *"What is the annual wellness stipend?"* | **Supported**: Answers `$1,500 USD` with citations: `[Acme_Corp_Policy_Handbook.pdf \| Page 1 \| Chunk 2]` |
| **Answerable (Security)** | *"What is the password requirement?"* | **Supported**: Answers `At least 16 characters with mandatory hardware MFA key` with citation for Page 2 |
| **Unanswerable (Out-of-domain)** | *"What is the CEO's stock grant?"* | **Guardrail Triggered**: Returns `"I don't know based on the provided documents."` with 0 citations |
| **Hallucination Trap** | *"Does Acme provide free private jet travel?"* | **Guardrail Triggered**: Refuses to speculate or invent details; returns guardrail fallback |

---

## 📁 Repository Structure

```
├── app.py              # Streamlit application UI and demo runner
├── rag_engine.py       # Core RAG engine: PDF parsing, chunking, FAISS, Gemini verification
├── requirements.txt    # Python dependencies
├── .gitignore          # Ignores venv, caches, indexes, and credentials
├── .env.example        # Environment variable template
└── README.md           # Documentation, architecture, and rubric mapping
```
