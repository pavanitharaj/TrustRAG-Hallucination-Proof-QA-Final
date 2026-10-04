"""
TrustRAG Engine - Hallucination-Proof RAG Core
=============================================
Architecture:
  PDF Upload -> Text Extraction (with page numbers) -> Semantic Chunking
  -> SentenceTransformer Embeddings -> FAISS Vector Search -> Top-k Retrieval
  -> Two-Stage Gemini Verification & Grounding
  -> Strict "I don't know" Guardrail
  -> Python-Generated Metadata Citations (LLM never generates citations)
"""

import os
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# PDF Extraction
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# Sentence Transformers for local free embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# FAISS for local fast vector index
try:
    import faiss
except ImportError:
    faiss = None

# Google GenAI SDK
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


@dataclass
class DocumentChunk:
    """Represents a discrete semantic chunk of a document with exact provenance."""
    doc_name: str
    page_number: int       # 1-indexed page number
    chunk_id: int          # Global or per-doc chunk sequence number
    text: str              # Cleaned text chunk
    char_count: int        # Character length

    def to_citation_label(self) -> str:
        """Standardized Python-generated citation identifier."""
        return f"[{self.doc_name} | Page {self.page_number} | Chunk {self.chunk_id}]"


@dataclass
class RetrievalResult:
    """Retrieved chunk along with similarity score."""
    chunk: DocumentChunk
    score: float           # Cosine similarity score (0.0 to 1.0)


@dataclass
class VerificationResult:
    """Result of Stage 1: Evidence Verification."""
    is_supported: bool
    reason: str
    supported_chunk_ids: List[int]


@dataclass
class RAGResponse:
    """Complete answer object with guardrail status and Python citations."""
    question: str
    answer: str
    is_supported: bool
    status_label: str       # "SUPPORTED" or "NOT_SUPPORTED"
    citations: List[Dict[str, Any]]
    retrieved_evidence: List[RetrievalResult]
    verification_reason: str


class TrustRAGEngine:
    """
    Hallucination-proof RAG engine enforcing strict grounding and verification.
    """

    GUARDRAIL_FALLBACK = "I don't know based on the provided documents."

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        gemini_model: str = "gemini-2.5-flash",
        min_relevance_threshold: float = 0.25,
    ):
        """
        Initialize the TrustRAG engine.

        Args:
            embedding_model_name: Local SentenceTransformer model (free, open source).
            chunk_size: Target characters per chunk (~100-120 words).
            chunk_overlap: Characters of overlap to preserve boundary context.
            gemini_model: Gemini model name for verification and answer generation.
            min_relevance_threshold: Minimum cosine similarity score for retrieval.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.gemini_model = gemini_model
        self.min_relevance_threshold = min_relevance_threshold

        # Initialize Embedding model lazily or immediately
        self._embedder = None
        self._embedding_model_name = embedding_model_name

        # Storage for indexed documents
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None
        self.indexed_files: List[Dict[str, Any]] = []

        # Gemini Client
        self._client = None

    @property
    def embedder(self):
        """Lazy load SentenceTransformer embedder."""
        if self._embedder is None:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers is not installed. Run: pip install sentence-transformers"
                )
            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder

    @property
    def genai_client(self):
        """Initialize Google GenAI client from GEMINI_API_KEY environment variable."""
        if self._client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key or api_key == "MY_GEMINI_API_KEY":
                raise ValueError(
                    "GEMINI_API_KEY is not set in the environment or .env file. "
                    "Please set GEMINI_API_KEY to a valid Google Gemini API key."
                )
            if genai is None:
                raise ImportError("google-genai is not installed. Run: pip install google-genai")
            self._client = genai.Client(api_key=api_key)
        return self._client

    # =========================================================================
    # Step 1: Text Extraction with Page Numbers
    # =========================================================================
    def extract_text_from_pdf(self, pdf_file, doc_name: str) -> List[Tuple[int, str]]:
        """
        Extract text from a PDF file preserving per-page tracking.

        Args:
            pdf_file: File-like object or file path.
            doc_name: Name of the document.

        Returns:
            List of (page_number_1_indexed, page_text) tuples.
        """
        if PdfReader is None:
            raise ImportError("pypdf is not installed. Run: pip install pypdf")

        reader = PdfReader(pdf_file)
        pages_data: List[Tuple[int, str]] = []

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            raw_text = page.extract_text() or ""
            # Clean text: normalize whitespace while keeping punctuation
            cleaned_text = re.sub(r"\s+", " ", raw_text).strip()
            if cleaned_text:
                pages_data.append((page_num, cleaned_text))

        return pages_data

    # =========================================================================
    # Step 2: Semantic Chunking (Sensible Size & Overlap)
    # =========================================================================
    def chunk_page_text(
        self,
        doc_name: str,
        page_num: int,
        page_text: str,
        starting_chunk_id: int
    ) -> List[DocumentChunk]:
        """
        Split a single page's text into overlapping chunks.
        Chunking is performed strictly within page boundaries so that every
        chunk has an exact, unequivocal page citation.
        """
        chunks: List[DocumentChunk] = []
        text_len = len(page_text)
        current_chunk_id = starting_chunk_id

        if text_len == 0:
            return chunks

        # If page text is smaller than chunk size, keep as single chunk
        if text_len <= self.chunk_size:
            chunk = DocumentChunk(
                doc_name=doc_name,
                page_number=page_num,
                chunk_id=current_chunk_id,
                text=page_text,
                char_count=text_len,
            )
            return [chunk]

        # Slide window with overlap, respecting sentence boundaries where possible
        start = 0
        while start < text_len:
            end = start + self.chunk_size

            # If not at the end of the text, look for sentence boundary near end
            if end < text_len:
                # Look for sentence ending within last 20% of the chunk window
                search_sub = page_text[max(start, end - 80): min(text_len, end + 40)]
                sentence_match = [m.end() for m in re.finditer(r"[.!?]\s+", search_sub)]
                if sentence_match:
                    # Adjust end to the closest sentence punctuation
                    adjusted_offset = (max(start, end - 80)) + sentence_match[-1]
                    if start < adjusted_offset <= text_len:
                        end = adjusted_offset

            chunk_text = page_text[start:end].strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        doc_name=doc_name,
                        page_number=page_num,
                        chunk_id=current_chunk_id,
                        text=chunk_text,
                        char_count=len(chunk_text),
                    )
                )
                current_chunk_id += 1

            # Step forward by chunk_size - chunk_overlap
            step = max(self.chunk_size - self.chunk_overlap, 100)
            start += step

        return chunks

    # =========================================================================
    # Step 3 & 4: Embeddings & FAISS Vector Indexing
    # =========================================================================
    def build_index(self, pdf_inputs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        """
        Parse, chunk, embed, and index multiple PDF documents.

        Args:
            pdf_inputs: List of tuples (document_name, file_object_or_path)

        Returns:
            Dictionary summary of indexed statistics.
        """
        self.chunks.clear()
        self.indexed_files.clear()
        total_pages = 0
        global_chunk_id = 1

        for doc_name, file_obj in pdf_inputs:
            pages = self.extract_text_from_pdf(file_obj, doc_name)
            doc_chunk_count = 0
            for page_num, text in pages:
                page_chunks = self.chunk_page_text(doc_name, page_num, text, global_chunk_id)
                self.chunks.extend(page_chunks)
                global_chunk_id += len(page_chunks)
                doc_chunk_count += len(page_chunks)

            total_pages += len(pages)
            self.indexed_files.append({
                "doc_name": doc_name,
                "pages": len(pages),
                "chunks": doc_chunk_count,
            })

        if not self.chunks:
            self.embeddings = None
            self.faiss_index = None
            return {"num_documents": 0, "total_pages": 0, "total_chunks": 0}

        # Generate SentenceTransformer embeddings
        chunk_texts = [c.text for c in self.chunks]
        raw_embeddings = self.embedder.encode(
            chunk_texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # L2-normalize vectors for exact cosine similarity calculation
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.embeddings = (raw_embeddings / norms).astype(np.float32)

        # Store in FAISS IndexFlatIP (Inner Product on normalized vectors = Cosine Similarity)
        dimension = self.embeddings.shape[1]
        if faiss is not None:
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(self.embeddings)
        else:
            self.faiss_index = None

        return {
            "num_documents": len(self.indexed_files),
            "total_pages": total_pages,
            "total_chunks": len(self.chunks),
            "documents": self.indexed_files,
        }

    # =========================================================================
    # Step 5: Top-k Evidence Retrieval
    # =========================================================================
    def retrieve(self, query: str, top_k: int = 4) -> List[RetrievalResult]:
        """
        Retrieve the top-k most relevant chunks for a question with cosine similarity.
        """
        if not self.chunks or self.embeddings is None:
            return []

        # Embed query and normalize
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        q_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1e-10
        query_embedding = (query_embedding / q_norm).astype(np.float32)

        results: List[RetrievalResult] = []

        if self.faiss_index is not None:
            # Query FAISS index
            k = min(top_k, len(self.chunks))
            distances, indices = self.faiss_index.search(query_embedding, k)
            for score, idx in zip(distances[0], indices[0]):
                if idx < len(self.chunks):
                    # Clip score to [0.0, 1.0] for clean reporting
                    clamped_score = max(0.0, min(1.0, float(score)))
                    results.append(RetrievalResult(chunk=self.chunks[idx], score=clamped_score))
        else:
            # Fallback pure NumPy dot product (guarantees runnability anywhere)
            scores = np.dot(self.embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            for idx in top_indices:
                score = max(0.0, min(1.0, float(scores[idx])))
                results.append(RetrievalResult(chunk=self.chunks[idx], score=score))

        return results

    # =========================================================================
    # Step 6 & 7: Evidence Verification & Guardrail
    # =========================================================================
    def verify_evidence(
        self,
        question: str,
        retrieved_results: List[RetrievalResult]
    ) -> VerificationResult:
        """
        Stage 1: Rigorous evidence verification step.
        Instructs Gemini to evaluate if the retrieved chunks directly and
        sufficiently contain the factual answer to the question.
        """
        if not retrieved_results:
            return VerificationResult(
                is_supported=False,
                reason="No document chunks were retrieved.",
                supported_chunk_ids=[],
            )

        # Quick guard: if top retrieved chunk has very low similarity, fail early
        top_score = retrieved_results[0].score
        if top_score < self.min_relevance_threshold:
            return VerificationResult(
                is_supported=False,
                reason=f"Top retrieved chunk relevance ({top_score:.2f}) is below minimum threshold ({self.min_relevance_threshold:.2f}).",
                supported_chunk_ids=[],
            )

        # Format context for verification with chunk metadata
        context_blocks = []
        for r in retrieved_results:
            context_blocks.append(
                f"[Chunk {r.chunk.chunk_id} | Document: '{r.chunk.doc_name}' | Page {r.chunk.page_number}]\n"
                f"{r.chunk.text}"
            )
        formatted_context = "\n\n".join(context_blocks)

        verification_prompt = f"""You are a strict, hallucination-proof factual verification auditor.
Your job is to determine whether the provided context contains direct, unequivocal, and sufficient factual evidence to answer the user's question.

CRITICAL RULES:
1. If the context does NOT contain the direct answer, you MUST set is_supported to false.
2. If answering the question requires speculation, outside knowledge, or guessing, you MUST set is_supported to false.
3. If the context only vaguely mentions related topics but does NOT state the specific fact requested, set is_supported to false.
4. If is_supported is true, list the exact chunk_ids that directly contain the proof.

User Question:
"{question}"

Retrieved Context Chunks:
{formatted_context}

Respond in strictly valid JSON format with keys:
{{
  "is_supported": true or false,
  "reason": "Clear 1-2 sentence explanation of why the evidence is or is not sufficient",
  "supported_chunk_ids": [list of integer chunk_ids that contain the proof]
}}
"""

        try:
            client = self.genai_client
            response = client.models.generate_content(
                model=self.gemini_model,
                contents=verification_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,  # Zero temperature for deterministic verification
                ),
            )

            response_text = response.text or "{}"
            parsed = json.loads(response_text)

            is_supported = bool(parsed.get("is_supported", False))
            reason = str(parsed.get("reason", "Verification evaluated context sufficiency."))
            supported_ids = [int(cid) for cid in parsed.get("supported_chunk_ids", [])]

            return VerificationResult(
                is_supported=is_supported,
                reason=reason,
                supported_chunk_ids=supported_ids,
            )

        except Exception as e:
            # If verification call fails or JSON fails to parse, default to safe refusal
            return VerificationResult(
                is_supported=False,
                reason=f"Verification safeguard triggered due to verification error: {str(e)}",
                supported_chunk_ids=[],
            )

    # =========================================================================
    # Step 8: Grounded Answer & Python-Generated Citations
    # =========================================================================
    def generate_grounded_answer(
        self,
        question: str,
        retrieved_results: List[RetrievalResult],
        verification: VerificationResult
    ) -> RAGResponse:
        """
        Generate answer if supported, or trigger strict 'I don't know' guardrail.
        Generates Python citations from metadata (the LLM is prohibited from citing).
        """
        # 1. Unanswerable Guardrail Check
        if not verification.is_supported:
            return RAGResponse(
                question=question,
                answer=self.GUARDRAIL_FALLBACK,
                is_supported=False,
                status_label="NOT_SUPPORTED",
                citations=[],
                retrieved_evidence=retrieved_results,
                verification_reason=verification.reason,
            )

        # Filter chunks that were verified as supporting the answer
        active_chunks = [
            r for r in retrieved_results
            if r.chunk.chunk_id in verification.supported_chunk_ids
        ]
        # If model marked supported but empty id list, use all retrieved chunks
        if not active_chunks:
            active_chunks = retrieved_results

        # Prepare verified factual context
        context_snippets = []
        for r in active_chunks:
            context_snippets.append(
                f"[Chunk ID: {r.chunk.chunk_id}]\n{r.chunk.text}"
            )
        context_str = "\n\n".join(context_snippets)

        generation_prompt = f"""You are the answering engine for TrustRAG.
Answer the user's question using ONLY the provided verified context facts.

STRICT INSTRUCTIONS:
- Answer faithfully and accurately based ONLY on the text below.
- Do NOT bring in external knowledge.
- Do NOT fabricate or extrapolate beyond what is written.
- IMPORTANT: DO NOT write citations, footnotes, or references like '[Document 1, Page 2]' in your answer.
  Citations are generated strictly and programmatically by Python after this step. Just provide the clear factual answer.

Question:
{question}

Verified Context:
{context_str}
"""

        try:
            client = self.genai_client
            response = client.models.generate_content(
                model=self.gemini_model,
                contents=generation_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,  # Low temperature for factual fidelity
                ),
            )
            raw_answer = (response.text or "").strip()
            if not raw_answer:
                raw_answer = self.GUARDRAIL_FALLBACK
                is_supported = False
            else:
                is_supported = True

        except Exception as e:
            raw_answer = f"{self.GUARDRAIL_FALLBACK} (Model generation error: {str(e)})"
            is_supported = False

        # 2. Programmatically generate citations in Python from metadata
        citations: List[Dict[str, Any]] = []
        if is_supported:
            for r in active_chunks:
                # Python-generated citation metadata
                citations.append({
                    "doc_name": r.chunk.doc_name,
                    "page_number": r.chunk.page_number,
                    "chunk_id": r.chunk.chunk_id,
                    "relevance_score": round(r.score * 100, 1),
                    "citation_label": r.chunk.to_citation_label(),
                    "excerpt": r.chunk.text[:220] + "..." if len(r.chunk.text) > 220 else r.chunk.text,
                    "full_text": r.chunk.text,
                })

        return RAGResponse(
            question=question,
            answer=raw_answer,
            is_supported=is_supported,
            status_label="SUPPORTED" if is_supported else "NOT_SUPPORTED",
            citations=citations,
            retrieved_evidence=retrieved_results,
            verification_reason=verification.reason,
        )

    # =========================================================================
    # High-level Pipeline Execution
    # =========================================================================
    def ask(self, question: str, top_k: int = 4) -> RAGResponse:
        """
        Execute full RAG pipeline:
        Retrieve -> Verify -> Answer -> Cite
        """
        if not self.chunks:
            return RAGResponse(
                question=question,
                answer="No documents have been uploaded and indexed yet. Please upload at least one PDF first.",
                is_supported=False,
                status_label="NOT_SUPPORTED",
                citations=[],
                retrieved_evidence=[],
                verification_reason="Knowledge base is empty.",
            )

        clean_question = question.strip()
        if not clean_question:
            return RAGResponse(
                question=question,
                answer="Please enter a valid question.",
                is_supported=False,
                status_label="NOT_SUPPORTED",
                citations=[],
                retrieved_evidence=[],
                verification_reason="Empty question provided.",
            )

        # 1. Retrieve
        retrieved_results = self.retrieve(clean_question, top_k=top_k)

        # 2. Verify
        verification = self.verify_evidence(clean_question, retrieved_results)

        # 3. Answer & 4. Cite (Deterministic in Python)
        return self.generate_grounded_answer(clean_question, retrieved_results, verification)
