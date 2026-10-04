import express from 'express';
import { createServer as createViteServer } from 'vite';
import dotenv from 'dotenv';
import { GoogleGenAI, Type } from '@google/genai';

dotenv.config();

const app = express();
const port = 3000;

app.use(express.json({ limit: '20mb' }));

// Initialize Google GenAI on the server
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

interface ChunkItem {
  id: number;
  docName: string;
  pageNumber: number;
  text: string;
  charCount: number;
}

interface RetrievedChunk {
  chunk: ChunkItem;
  score: number;
}

// Helper for resilient Gemini API calls with retry and model fallback
async function callGeminiWithFallback(
  callFn: (modelName: string) => Promise<any>,
  models: string[] = ['gemini-2.5-flash', 'gemini-3.8-flash', 'gemini-flash-latest']
) {
  let lastError: any = null;
  for (const model of models) {
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        return await callFn(model);
      } catch (err: any) {
        lastError = err;
        const msg = String(err?.message || '');
        if (msg.includes('503') || msg.includes('429') || msg.includes('UNAVAILABLE')) {
          // Wait briefly before retry
          await new Promise((r) => setTimeout(r, 600));
          continue;
        }
        break; // Other error, try next model
      }
    }
  }
  throw lastError;
}
const DEFAULT_DOCUMENTS: { name: string; pages: { pageNum: number; content: string }[] }[] = [
  {
    name: 'Acme_Corp_Policy_Handbook.pdf',
    pages: [
      {
        pageNum: 1,
        content: `Acme Corporation - Employee Handbook & Working Guidelines.
Section 1. Core Working Hours: All full-time employees are expected to be available between 10:00 AM and 4:00 PM Eastern Time for synchronous team coordination and meetings. Outside of these core hours, employees maintain full scheduling flexibility.
Section 2. Annual Wellness Reimbursement: Acme provides a generous annual wellness stipend of $1,500 USD per employee. Eligible expenses include fitness memberships, athletic footwear, personal training, meditation apps, and ergonomic home accessories.
Section 3. Home Equipment Setup: All newly hired remote employees receive an upfront one-time technology and ergonomics allowance of $750 to equip their home office with dual monitors or standing desk converters.`,
      },
      {
        pageNum: 2,
        content: `Acme Corporation - Security & Travel Guidelines.
Section 4. Cybersecurity & Authentication: Passwords must be a minimum of 16 characters in length, containing upper and lowercase letters, numbers, and symbols. Hardware security keys (FIDO2/WebAuthn) are mandatory for all production systems. Password sharing is grounds for immediate disciplinary review.
Section 5. Travel Expense Policy: Business travel meals are reimbursed up to a strict per-diem cap of $85 USD per day. Itemized receipts must be submitted through the finance portal within 14 calendar days of returning from the business trip.
Section 6. International Device Transport: Company-issued laptops must never be transported to or operated within countries categorized under Tier-3 national security travel warnings without prior written authorization from the VP of Information Security.`,
      },
    ],
  },
  {
    name: 'Apollo_CleanTech_Q3_Report.pdf',
    pages: [
      {
        pageNum: 1,
        content: `Apollo CleanTech Inc. Q3 Financial & Operational Summary.
Operational Highlights: Total solar energy generated across all operational commercial farms reached 420 gigawatt-hours in Q3, representing a 28% increase year-over-year. Capital expenditures for battery storage expansion totaled $14.2 million USD.
Revenue Breakdown: Commercial solar subscriptions contributed $38.6 million in quarterly recurring revenue, while residential microgrid installations delivered $12.1 million. Total consolidated revenue reached $50.7 million.`,
      },
      {
        pageNum: 2,
        content: `Apollo CleanTech Inc. Sustainability & Governance.
Carbon Offset Milestones: The operational grid displacement achieved by Apollo projects reduced greenhouse gas emissions by an estimated 185,000 metric tons of CO2 equivalent during the third quarter.
Board of Directors Update: Dr. Elena Vance was appointed Chairperson of the Audit and Risk Committee, effective September 15. The board confirmed the next quarterly dividend payment date as November 12.`,
      },
    ],
  },
];

// Helper: Semantic chunking respecting sentence boundaries
function chunkDocumentPages(
  docName: string,
  pages: { pageNum: number; content: string }[],
  chunkSize: number = 550,
  overlap: number = 100
): ChunkItem[] {
  const chunks: ChunkItem[] = [];
  let globalId = 1;

  for (const page of pages) {
    const text = page.content.replace(/\s+/g, ' ').trim();
    if (!text) continue;

    if (text.length <= chunkSize) {
      chunks.push({
        id: globalId++,
        docName,
        pageNumber: page.pageNum,
        text,
        charCount: text.length,
      });
      continue;
    }

    let start = 0;
    while (start < text.length) {
      let end = start + chunkSize;

      if (end < text.length) {
        // Look for sentence termination near end boundary
        const windowSub = text.substring(Math.max(start, end - 80), Math.min(text.length, end + 40));
        const periodIdx = windowSub.search(/[.!?]\s+/);
        if (periodIdx !== -1) {
          const adjusted = Math.max(start, end - 80) + periodIdx + 1;
          if (adjusted > start && adjusted <= text.length) {
            end = adjusted;
          }
        }
      }

      const chunkText = text.substring(start, end).trim();
      if (chunkText) {
        chunks.push({
          id: globalId++,
          docName,
          pageNumber: page.pageNum,
          text: chunkText,
          charCount: chunkText.length,
        });
      }

      const step = Math.max(chunkSize - overlap, 80);
      start += step;
    }
  }

  return chunks;
}

// Compute word n-gram & keyword similarity vector score
function computeRelevanceScore(query: string, text: string): number {
  const tokenize = (s: string) =>
    s
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter((w) => w.length > 2);

  const qTokens = tokenize(query);
  const docTokens = tokenize(text);

  if (qTokens.length === 0 || docTokens.length === 0) return 0;

  // Unigram match count
  const docSet = new Set(docTokens);
  let matches = 0;
  for (const t of qTokens) {
    if (docSet.has(t)) {
      matches += 1;
    } else {
      // Substring stem check
      for (const d of docSet) {
        if (d.includes(t) || t.includes(d)) {
          matches += 0.6;
          break;
        }
      }
    }
  }

  // Jaccard + Query coverage hybrid score
  const coverage = matches / qTokens.length;
  const termDensity = matches / Math.sqrt(docTokens.length);
  const combined = coverage * 0.75 + Math.min(termDensity, 1.0) * 0.25;

  return Math.max(0.05, Math.min(0.98, Number(combined.toFixed(3))));
}

// Top-k retrieval
function retrieveTopK(query: string, chunks: ChunkItem[], k: number = 4): RetrievedChunk[] {
  const scored = chunks.map((chunk) => ({
    chunk,
    score: computeRelevanceScore(query, chunk.text),
  }));

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

// REST API Route: Get default / demo documents
app.get('/api/trustrag/default-docs', (req, res) => {
  res.json({
    documents: DEFAULT_DOCUMENTS.map((d) => ({
      name: d.name,
      pageCount: d.pages.length,
      pages: d.pages,
    })),
  });
});

// REST API Route: Full RAG Query Execution
app.post('/api/trustrag/query', async (req, res) => {
  try {
    const { question, documents, topK = 4, minThreshold = 0.2 } = req.body;

    if (!question || typeof question !== 'string') {
      return res.status(400).json({ error: 'A valid question string is required.' });
    }

    // Prepare all chunks
    let allChunks: ChunkItem[] = [];
    const docsToProcess = documents && documents.length > 0 ? documents : DEFAULT_DOCUMENTS;

    for (const doc of docsToProcess) {
      const docChunks = chunkDocumentPages(doc.name, doc.pages, 550, 100);
      allChunks.push(...docChunks);
    }

    if (allChunks.length === 0) {
      return res.json({
        question,
        answer: "I don't know based on the provided documents.",
        isSupported: false,
        statusLabel: 'NOT_SUPPORTED',
        verificationReason: 'No documents or text chunks are indexed.',
        citations: [],
        retrievedEvidence: [],
      });
    }

    // 1. Retrieve top-k
    const retrieved = retrieveTopK(question, allChunks, topK);

    // Format context for verification
    const contextFormatted = retrieved
      .map(
        (r) =>
          `[Chunk ID: ${r.chunk.id} | Document: ${r.chunk.docName} | Page: ${r.chunk.pageNumber} | Score: ${(
            r.score * 100
          ).toFixed(1)}%]\n${r.chunk.text}`
      )
      .join('\n\n');

    // 2. Stage 1: Evidence Verification with Gemini
    const verificationPrompt = `You are a strict, hallucination-proof factual verification auditor.
Your job is to determine whether the provided context contains direct, unequivocal, and sufficient factual evidence to answer the user's question.

CRITICAL RULES:
1. If the context does NOT contain the direct answer, you MUST set is_supported to false.
2. If answering the question requires speculation, outside knowledge, or guessing, you MUST set is_supported to false.
3. If the context only vaguely mentions related topics but does NOT state the specific fact requested, set is_supported to false.
4. If is_supported is true, list the exact chunk_ids that directly contain the proof.

User Question:
"${question}"

Retrieved Context Chunks:
${contextFormatted}
`;

    let isSupported = false;
    let verificationReason = 'Context audited for factual sufficiency.';
    let supportedChunkIds: number[] = [];

    // Fallback if top score is extremely poor
    if (retrieved.length > 0 && retrieved[0].score < minThreshold) {
      isSupported = false;
      verificationReason = `Top retrieved chunk similarity (${(retrieved[0].score * 100).toFixed(
        1
      )}%) is below minimum confidence threshold (${(minThreshold * 100).toFixed(0)}%).`;
    } else {
      try {
        const verifyResponse = await callGeminiWithFallback((model) =>
          ai.models.generateContent({
            model,
            contents: verificationPrompt,
            config: {
              responseMimeType: 'application/json',
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  is_supported: {
                    type: Type.BOOLEAN,
                    description: 'Whether context directly and sufficiently proves the answer',
                  },
                  reason: {
                    type: Type.STRING,
                    description: '1-2 sentence explanation of proof sufficiency',
                  },
                  supported_chunk_ids: {
                    type: Type.ARRAY,
                    items: { type: Type.INTEGER },
                    description: 'IDs of chunks that contain the direct facts',
                  },
                },
                required: ['is_supported', 'reason'],
              },
              temperature: 0.0,
            },
          })
        );

        const parsed = JSON.parse(verifyResponse.text || '{}');
        isSupported = Boolean(parsed.is_supported);
        verificationReason = parsed.reason || verificationReason;
        supportedChunkIds = Array.isArray(parsed.supported_chunk_ids) ? parsed.supported_chunk_ids : [];
      } catch (err: any) {
        console.error('Verification auditor error:', err);
        // Fallback safeguard: if verification fails, refuse to guess
        isSupported = false;
        verificationReason = `Verification safeguard triggered: ${err.message || 'Auditor parsing failure'}`;
      }
    }

    // 3. Stage 2: Grounded Answer Generation OR Guardrail
    const GUARDRAIL_FALLBACK = "I don't know based on the provided documents.";
    let answerText = GUARDRAIL_FALLBACK;

    if (isSupported) {
      const activeEvidence =
        supportedChunkIds.length > 0
          ? retrieved.filter((r) => supportedChunkIds.includes(r.chunk.id))
          : retrieved;

      const verifiedContextText = activeEvidence
        .map((r) => `[Chunk ${r.chunk.id}]: ${r.chunk.text}`)
        .join('\n\n');

      const answerPrompt = `You are the answering engine for TrustRAG.
Answer the user's question using ONLY the provided verified context facts.

STRICT INSTRUCTIONS:
- Answer faithfully and accurately based ONLY on the text below.
- Do NOT bring in external knowledge.
- Do NOT fabricate or extrapolate beyond what is written.
- IMPORTANT: DO NOT write citations, footnotes, or references like '[Document 1, Page 2]' in your answer.
  Citations are generated strictly and programmatically by Python/code after this step. Just provide the clear factual answer.

Question:
${question}

Verified Context:
${verifiedContextText}
`;

      try {
        const genResponse = await callGeminiWithFallback((model) =>
          ai.models.generateContent({
            model,
            contents: answerPrompt,
            config: {
              temperature: 0.1,
            },
          })
        );

        answerText = (genResponse.text || '').trim();
        if (!answerText) {
          answerText = GUARDRAIL_FALLBACK;
          isSupported = false;
        }
      } catch (err: any) {
        console.error('Answer generation error:', err);
        answerText = GUARDRAIL_FALLBACK;
        isSupported = false;
        verificationReason = `Generation safeguard: ${err.message}`;
      }
    }

    // 4. Deterministic Code Citations from metadata (LLM never invents citations)
    const citations = isSupported
      ? (supportedChunkIds.length > 0
          ? retrieved.filter((r) => supportedChunkIds.includes(r.chunk.id))
          : retrieved
        ).map((r) => ({
          docName: r.chunk.docName,
          pageNumber: r.chunk.pageNumber,
          chunkId: r.chunk.id,
          relevanceScore: Number((r.score * 100).toFixed(1)),
          citationLabel: `[${r.chunk.docName} | Page ${r.chunk.pageNumber} | Chunk #${r.chunk.id}]`,
          excerpt:
            r.chunk.text.length > 220 ? r.chunk.text.substring(0, 220) + '...' : r.chunk.text,
          fullText: r.chunk.text,
        }))
      : [];

    return res.json({
      question,
      answer: answerText,
      isSupported,
      statusLabel: isSupported ? 'SUPPORTED' : 'NOT_SUPPORTED',
      verificationReason,
      citations,
      retrievedEvidence: retrieved.map((r) => ({
        chunkId: r.chunk.id,
        docName: r.chunk.docName,
        pageNumber: r.chunk.pageNumber,
        score: Number((r.score * 100).toFixed(1)),
        text: r.chunk.text,
      })),
      totalIndexedChunks: allChunks.length,
    });
  } catch (globalErr: any) {
    console.error('TrustRAG query failure:', globalErr);
    res.status(500).json({
      error: globalErr.message || 'Internal Server Error during RAG execution',
    });
  }
});

// Setup Vite Dev middleware or Static serving
async function startServer() {
  if (process.env.NODE_ENV === 'production') {
    app.use(express.static('dist'));
    app.get('*', (req, res) => {
      res.sendFile('index.html', { root: 'dist' });
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(port, '0.0.0.0', () => {
    console.log(`TrustRAG server running on http://0.0.0.0:${port}`);
  });
}

startServer();
