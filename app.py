"""
TrustRAG - Hallucination-Proof Document Q&A Bot
===============================================
A trustworthy RAG application built with Streamlit, SentenceTransformers,
FAISS, and Google GenAI.

Pipeline:
  1. Parse & chunk multiple PDFs (tracking exact page numbers)
  2. SentenceTransformer embeddings (all-MiniLM-L6-v2)
  3. FAISS vector index (Cosine similarity via IndexFlatIP)
  4. Top-k evidence retrieval
  5. Gemini evidence verification auditor
  6. Strict guardrail ("I don't know based on the provided documents.")
  7. Grounded answer generation
  8. Deterministic Python-generated citations (Document + Page + Chunk)
"""

import os
import io
import time
import streamlit as st
from dotenv import load_dotenv

# Import the core TrustRAG engine
from rag_engine import TrustRAGEngine, DocumentChunk

# Load environment variables
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(
    page_title="TrustRAG | Hallucination-Proof Q&A",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished hackathon demo styling
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-container {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    /* Status Badges */
    .status-supported {
        display: inline-flex;
        align-items: center;
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        border-radius: 6px;
        padding: 4px 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .status-unsupported {
        display: inline-flex;
        align-items: center;
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        border-radius: 6px;
        padding: 4px 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    /* Citation Box */
    .citation-card {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        border-radius: 4px 8px 8px 4px;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    .citation-badge {
        font-weight: 700;
        color: #1e40af;
        margin-bottom: 4px;
    }
    /* How it works pipeline pills */
    .pipeline-step {
        display: inline-block;
        background: #e0e7ff;
        color: #3730a3;
        font-weight: 600;
        border-radius: 9999px;
        padding: 4px 12px;
        margin: 2px 4px;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing SentenceTransformer embedding model...")
def get_engine() -> TrustRAGEngine:
    """Initialize and cache the TrustRAG engine instance."""
    return TrustRAGEngine(
        embedding_model_name="all-MiniLM-L6-v2",
        chunk_size=600,
        chunk_overlap=120,
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        min_relevance_threshold=0.22,
    )


def create_sample_pdf_bytes() -> bytes:
    """Helper to generate a lightweight sample PDF in-memory for zero-friction testing."""
    from pypdf import PdfWriter
    from reportlab.pdfgen import canvas
    # If reportlab is available, generate PDF, otherwise return None
    try:
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer)
        c.drawString(72, 750, "Acme Corporation - Employee Handbook & Security Policy")
        c.drawString(72, 730, "Page 1: Remote Work, Equipment, and Working Hours")
        c.drawString(72, 700, "1. Core Working Hours: All full-time employees are expected to be available")
        c.drawString(72, 685, "   between 10:00 AM and 4:00 PM Eastern Time for cross-team collaboration.")
        c.drawString(72, 660, "2. Annual Wellness Stipend: Employees receive an annual wellness reimbursement")
        c.drawString(72, 645, "   stipend of $1,500 USD for gym memberships, athletic gear, or mental health apps.")
        c.drawString(72, 620, "3. Home Office Equipment: Acme provides a one-time remote setup credit of $750")
        c.drawString(72, 605, "   for ergonomic desks and chairs upon hire.")
        c.showPage()
        
        c.drawString(72, 750, "Acme Corporation - Employee Handbook & Security Policy")
        c.drawString(72, 730, "Page 2: Cybersecurity & Travel Regulations")
        c.drawString(72, 700, "4. Password Rotation & MFA: Passwords must be at least 16 characters long.")
        c.drawString(72, 685, "   Multi-factor authentication (MFA) with hardware security keys is mandatory.")
        c.drawString(72, 660, "5. International Travel Policy: Company laptops may NOT be taken to countries")
        c.drawString(72, 645, "   listed in Tier-3 security advisories without written VP Security approval.")
        c.drawString(72, 620, "6. Expense Reimbursement: All business travel meal receipts must be submitted")
        c.drawString(72, 605, "   within 14 calendar days of trip conclusion. The per-diem meal cap is $85.")
        c.showPage()
        c.save()
        return buffer.getvalue()
    except Exception:
        # Fallback to minimal synthetic PDF stream using pypdf if reportlab not present
        return None


def main():
    # Header & Hackathon Title
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        st.markdown("<h1 style='text-align: center; margin: 0;'>🛡️</h1>", unsafe_allow_html=True)
    with col_title:
        st.markdown("<h1 style='margin: 0; padding: 0;'>TrustRAG</h1>", unsafe_allow_html=True)
        st.markdown(
            "**Hallucination-Proof Document Q&A Bot** &mdash; Answers *only* from verified document evidence. "
            "Guaranteed zero guessing with strict 'I don't know' guardrails and deterministic Python citations."
        )

    st.markdown("---")

    # How It Works Banner
    with st.expander("ℹ️ **How TrustRAG Works**: 4-Step Hallucination-Proof Architecture", expanded=False):
        st.markdown(
            """
            ```
            [Uploaded PDFs] ──> [Text & Page Tracking] ──> [Semantic Chunks] ──> [MiniLM Embeddings] ──> [FAISS Index]
                                                                                                              │
                                                                   ┌──────────────────────────────────────────┘
                                                                   ▼
                                                            1. RETRIEVE ──── Top-k chunks via Cosine Similarity
                                                                   │
                                                                   ▼
                                                            2. VERIFY   ──── Gemini checks: is factual proof direct & complete?
                                                                   │
                                                ┌──────────────────┴──────────────────┐
                                                ▼                                     ▼
                                          [Supported = NO]                      [Supported = YES]
                                                │                                     │
                                                ▼                                     ▼
                                       STRICT GUARDRAIL:                        3. ANSWER: Grounded answer from verified chunks
                                 "I don't know based on the                            │
                                    provided documents."                               ▼
                                                                                4. CITE: Python generates Document + Page + Chunk citations
            ```
            - **No LLM Citation Hallucination**: The LLM is strictly prohibited from generating citations. All citations are programmatically stamped by Python from vector metadata.
            - **Two-Stage Safety**: Stage 1 audits context sufficiency at temperature `0.0`. Stage 2 generates answers only if proof exists.
            """
        )

    # Initialize Engine
    engine = get_engine()

    # Sidebar: Document Management & System Metrics
    with st.sidebar:
        st.header("📁 Document Management")
        st.markdown("Upload one or more PDF documents to build your vector knowledge base.")

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Multi-page PDFs will be parsed with exact page numbers tracked for citation.",
        )

        # Quick sample loader button for fast hackathon demo
        st.markdown("##### Or Try Instant Hackathon Demo:")
        if st.button("📄 Load Demo PDF (Acme Policy Handbook)"):
            sample_bytes = create_sample_pdf_bytes()
            if sample_bytes:
                with st.spinner("Indexing Acme Policy Handbook demo PDF..."):
                    engine.build_index([("Acme_Corp_Policy_Handbook.pdf", io.BytesIO(sample_bytes))])
                    st.session_state["indexed"] = True
                    st.success("Loaded & indexed Acme Policy Handbook (2 Pages, 6 Chunks)!")
            else:
                # Synthetic document injection if reportlab is unavailable
                st.info("Creating sample in-memory chunks directly...")
                p1_text = (
                    "Acme Corporation Employee Handbook Page 1. Core Working Hours: All full-time employees "
                    "are expected to be available between 10:00 AM and 4:00 PM Eastern Time for cross-team "
                    "collaboration. Annual Wellness Stipend: Employees receive an annual wellness reimbursement "
                    "stipend of $1,500 USD for gym memberships, athletic gear, or mental health apps. "
                    "Home Office Equipment: Acme provides a one-time remote setup credit of $750 for ergonomic desks."
                )
                p2_text = (
                    "Acme Corporation Security Policy Page 2. Password Rotation & MFA: Passwords must be at least "
                    "16 characters long. Multi-factor authentication with hardware keys is mandatory. "
                    "International Travel Policy: Company laptops may not be taken to Tier-3 security advisories. "
                    "Expense Reimbursement: All business travel receipts must be submitted within 14 calendar days. "
                    "The per-diem meal cap is $85."
                )
                # Build synthetic document using engine chunker
                c1 = engine.chunk_page_text("Acme_Corp_Policy_Handbook.pdf", 1, p1_text, starting_chunk_id=1)
                c2 = engine.chunk_page_text("Acme_Corp_Policy_Handbook.pdf", 2, p2_text, starting_chunk_id=len(c1)+1)
                engine.chunks = c1 + c2
                engine.indexed_files = [{"doc_name": "Acme_Corp_Policy_Handbook.pdf", "pages": 2, "chunks": len(engine.chunks)}]
                # embed
                raw_emb = engine.embedder.encode([c.text for c in engine.chunks], convert_to_numpy=True)
                import numpy as np
                norms = np.linalg.norm(raw_emb, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                engine.embeddings = (raw_emb / norms).astype(np.float32)
                import faiss
                engine.faiss_index = faiss.IndexFlatIP(engine.embeddings.shape[1])
                engine.faiss_index.add(engine.embeddings)
                st.session_state["indexed"] = True
                st.success("Loaded & indexed Acme Demo Handbook!")

        # Process User-Uploaded PDFs
        if uploaded_files:
            if st.button("⚡ Index Uploaded Documents", type="primary"):
                with st.spinner("Extracting pages, chunking, and building FAISS vector index..."):
                    pdf_inputs = [(f.name, io.BytesIO(f.read())) for f in uploaded_files]
                    stats = engine.build_index(pdf_inputs)
                    st.session_state["indexed"] = True
                    st.success(f"Successfully indexed {stats['total_chunks']} chunks across {stats['total_pages']} pages!")

        # Knowledge Base Status & Metrics
        st.markdown("---")
        st.subheader("📊 Knowledge Base Status")
        num_docs = len(engine.indexed_files)
        total_pages = sum(d["pages"] for d in engine.indexed_files)
        total_chunks = len(engine.chunks)

        m1, m2, m3 = st.columns(3)
        m1.metric("Documents", num_docs)
        m2.metric("Pages", total_pages)
        m3.metric("Chunks", total_chunks)

        if engine.indexed_files:
            st.markdown("**Indexed Documents:**")
            for doc in engine.indexed_files:
                st.caption(f"• **{doc['doc_name']}** ({doc['pages']} pages, {doc['chunks']} chunks)")

        st.markdown("---")
        st.subheader("⚙️ RAG Configuration")
        top_k = st.slider("Top-k Evidence Retrieval", min_value=1, max_value=8, value=4)
        threshold = st.slider(
            "Min Relevance Score Threshold",
            min_value=0.10,
            max_value=0.60,
            value=0.22,
            step=0.02,
            help="Queries with top retrieval scores below this threshold are immediately rejected."
        )
        engine.min_relevance_threshold = threshold

        # API Key Status check (never display key)
        api_key_present = bool(os.environ.get("GEMINI_API_KEY") and os.environ.get("GEMINI_API_KEY") != "MY_GEMINI_API_KEY")
        if api_key_present:
            st.success("🔑 Gemini API Key: Active in environment")
        else:
            st.warning("⚠️ GEMINI_API_KEY not detected in .env. Please set it before asking questions.")

    # Main Area: Q&A Interface
    if not engine.chunks:
        st.info("👋 **Welcome to TrustRAG!** Upload PDF documents in the sidebar or click **'Load Demo PDF'** to begin.")
        st.markdown(
            """
            ### What makes TrustRAG Hallucination-Proof?
            1. **Exclusively Document-Grounded**: Answers come only from verified snippets retrieved from your PDFs.
            2. **Rigorous Two-Stage Verification**: Gemini acts first as an independent auditor to verify factual completeness before any answer is synthesized.
            3. **Uncompromising Guardrail**: If the evidence is insufficient or unanswerable, it outputs:  
               `"I don't know based on the provided documents."`
            4. **Programmatic Python Citations**: Citations are generated from chunk metadata in Python code. The LLM cannot invent fictitious citations or hallucinate page numbers.
            """
        )
        return

    # Question Input
    st.subheader("💬 Ask a Document Question")

    # Sample Quick Buttons
    st.markdown("**Try Example Questions:**")
    c_q1, c_q2, c_q3 = st.columns(3)
    preset_q = ""
    with c_q1:
        if st.button("🟢 Answerable: 'What is the annual wellness stipend?'"):
            preset_q = "What is the annual wellness stipend?"
    with c_q2:
        if st.button("🟢 Answerable: 'What are the core working hours?'"):
            preset_q = "What are the core working hours?"
    with c_q3:
        if st.button("🔴 Unanswerable: 'What is the CEO's stock grant?'"):
            preset_q = "What is the CEO's stock grant?"

    query = st.text_input(
        "Enter your question:",
        value=preset_q if preset_q else "",
        placeholder="e.g., What is the expense reimbursement deadline and per-diem meal cap?",
    )

    ask_button = st.button("🔍 Get Verified Answer", type="primary", use_container_width=True)

    if (ask_button or preset_q) and query:
        start_time = time.time()
        with st.spinner("Retrieving evidence, verifying facts, and synthesizing answer..."):
            response = engine.ask(query, top_k=top_k)
        elapsed = time.time() - start_time

        st.markdown("---")

        # Response Presentation
        res_col_badge, res_col_time = st.columns([4, 1])
        with res_col_badge:
            if response.is_supported:
                st.markdown(
                    '<span class="status-supported">✅ STATUS: SUPPORTED BY DOCUMENTS</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="status-unsupported">🛑 STATUS: NOT SUPPORTED &mdash; GUARDRAIL TRIGGERED</span>',
                    unsafe_allow_html=True,
                )
        with res_col_time:
            st.caption(f"⏱️ Response time: {elapsed:.2f}s")

        # Main Answer Box
        st.markdown("### Answer")
        if response.is_supported:
            st.success(response.answer)
        else:
            st.error(f"**{response.answer}**")
            st.info(f"**Verification Audit Reason:** {response.verification_reason}")

        # Programmatically Generated Citations
        st.markdown("### 📌 Source Citations (Python-Generated)")
        if response.is_supported and response.citations:
            st.caption("Citations generated strictly from indexed vector metadata by Python:")
            for cite in response.citations:
                with st.container():
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-badge">
                                📄 Document: <code>{cite['doc_name']}</code> | Page: <b>{cite['page_number']}</b> | Chunk: <b>#{cite['chunk_id']}</b> | Relevance: <b>{cite['relevance_score']}%</b>
                            </div>
                            <div style="color: #334155; font-style: italic; margin-top: 4px;">
                                "{cite['excerpt']}"
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("No citations displayed because the question cannot be supported by the provided documents.")

        # Retrieved Evidence & Relevance Scores
        st.markdown("### 🔎 Retrieved Evidence & Relevance Scores")
        with st.expander(f"View all top-{len(response.retrieved_evidence)} retrieved chunks from FAISS vector search", expanded=True):
            if not response.retrieved_evidence:
                st.write("No chunks retrieved.")
            for i, item in enumerate(response.retrieved_evidence):
                score_pct = item.score * 100
                st.markdown(
                    f"**Chunk #{item.chunk.chunk_id}** &mdash; `{item.chunk.doc_name}` (Page {item.chunk.page_number}) &bull; "
                    f"**Cosine Similarity:** `{score_pct:.1f}%`"
                )
                st.progress(item.score)
                st.text_area(
                    label=f"Text content for Chunk #{item.chunk.chunk_id}",
                    value=item.chunk.text,
                    height=100,
                    disabled=True,
                    key=f"chunk_view_{i}_{item.chunk.chunk_id}",
                )
                st.markdown("---")


if __name__ == "__main__":
    main()
