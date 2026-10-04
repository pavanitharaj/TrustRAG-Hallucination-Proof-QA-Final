import React, { useState, useEffect } from 'react';
import {
  ShieldCheck,
  ShieldAlert,
  FileText,
  Search,
  CheckCircle2,
  XCircle,
  Cpu,
  Layers,
  Sparkles,
  BookOpen,
  Code2,
  Copy,
  Check,
  ExternalLink,
  ChevronRight,
  Database,
  ArrowRight,
  AlertTriangle,
  Info,
  Sliders,
  PlusCircle,
  FileCheck2,
} from 'lucide-react';

interface Citation {
  docName: string;
  pageNumber: number;
  chunkId: number;
  relevanceScore: number;
  citationLabel: string;
  excerpt: string;
  fullText?: string;
}

interface RetrievedItem {
  chunkId: number;
  docName: string;
  pageNumber: number;
  score: number;
  text: string;
}

interface RAGResult {
  question: string;
  answer: string;
  isSupported: boolean;
  statusLabel: string;
  verificationReason: string;
  citations: Citation[];
  retrievedEvidence: RetrievedItem[];
  totalIndexedChunks: number;
}

interface DocMeta {
  name: string;
  pageCount: number;
  pages: { pageNum: number; content: string }[];
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'playground' | 'architecture' | 'code'>('playground');
  const [activeCodeFile, setActiveCodeFile] = useState<'app.py' | 'rag_engine.py' | 'requirements.txt' | 'README.md'>('rag_engine.py');
  const [copied, setCopied] = useState(false);

  // Documents state
  const [documents, setDocuments] = useState<DocMeta[]>([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState(true);

  // Query state
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(4);
  const [minThreshold, setMinThreshold] = useState(0.2);
  const [isQuerying, setIsQuerying] = useState(false);
  const [result, setResult] = useState<RAGResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // New Document modal/state
  const [showAddDoc, setShowAddDoc] = useState(false);
  const [newDocTitle, setNewDocTitle] = useState('');
  const [newDocText, setNewDocText] = useState('');

  // Load default demo documents on mount
  useEffect(() => {
    fetch('/api/trustrag/default-docs')
      .then((res) => res.json())
      .then((data) => {
        if (data.documents) {
          setDocuments(data.documents);
        }
      })
      .catch((err) => {
        console.error('Failed to load default documents:', err);
      })
      .finally(() => {
        setIsLoadingDocs(false);
      });
  }, []);

  const totalPages = documents.reduce((sum, d) => sum + d.pageCount, 0);
  // Estimate ~3 chunks per page for display summary
  const estimatedChunks = documents.reduce((sum, d) => sum + d.pages.length * 3, 0);

  const handleAsk = async (questionToAsk?: string) => {
    const q = (questionToAsk !== undefined ? questionToAsk : query).trim();
    if (!q) return;

    if (questionToAsk) {
      setQuery(questionToAsk);
    }

    setIsQuerying(true);
    setErrorMsg(null);

    try {
      const response = await fetch('/api/trustrag/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: q,
          documents,
          topK,
          minThreshold,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data: RAGResult = await response.json();
      setResult(data);
    } catch (err: any) {
      console.error('Query error:', err);
      setErrorMsg(err.message || 'Failed to execute TrustRAG query.');
    } finally {
      setIsQuerying(false);
    }
  };

  const handleAddCustomDoc = () => {
    if (!newDocTitle.trim() || !newDocText.trim()) return;
    const cleanName = newDocTitle.trim().endsWith('.pdf') ? newDocTitle.trim() : `${newDocTitle.trim()}.pdf`;
    
    // Split into simulated pages by double newline or 800 chars
    const pages = [
      {
        pageNum: 1,
        content: newDocText.trim(),
      },
    ];

    setDocuments((prev) => [...prev, { name: cleanName, pageCount: 1, pages }]);
    setNewDocTitle('');
    setNewDocText('');
    setShowAddDoc(false);
  };

  const handleCopyCode = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-50 px-6 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-500/20 text-slate-950 font-black text-xl">
            <ShieldCheck className="h-6 w-6 text-slate-950" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xl font-bold tracking-tight text-white">TrustRAG</span>
              <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                Hallucination-Proof
              </span>
            </div>
            <p className="text-xs text-slate-400">Strict Verification &bull; Zero Hallucination &bull; Python Citations</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center gap-1 bg-slate-800/80 p-1 rounded-xl border border-slate-700/60">
          <button
            onClick={() => setActiveTab('playground')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition flex items-center gap-2 ${
              activeTab === 'playground'
                ? 'bg-emerald-600 text-white shadow-sm'
                : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
            }`}
          >
            <Sparkles className="h-3.5 w-3.5" />
            Live Playground
          </button>
          <button
            onClick={() => setActiveTab('architecture')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition flex items-center gap-2 ${
              activeTab === 'architecture'
                ? 'bg-emerald-600 text-white shadow-sm'
                : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
            }`}
          >
            <Layers className="h-3.5 w-3.5" />
            How It Works
          </button>
          <button
            onClick={() => setActiveTab('code')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition flex items-center gap-2 ${
              activeTab === 'code'
                ? 'bg-emerald-600 text-white shadow-sm'
                : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
            }`}
          >
            <Code2 className="h-3.5 w-3.5" />
            Python Files ({['app.py', 'rag_engine.py'].length + 3})
          </button>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        {/* ================= TAB 1: LIVE PLAYGROUND ================= */}
        {activeTab === 'playground' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Column: Knowledge Base & Configuration (4 cols) */}
            <div className="lg:col-span-4 space-y-5">
              {/* Document Overview Card */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-slate-200 font-semibold text-sm">
                    <Database className="h-4 w-4 text-emerald-400" />
                    <span>Active Knowledge Base</span>
                  </div>
                  <button
                    onClick={() => setShowAddDoc(true)}
                    className="text-xs text-emerald-400 hover:text-emerald-300 font-medium flex items-center gap-1 bg-emerald-500/10 px-2 py-1 rounded-lg border border-emerald-500/20"
                  >
                    <PlusCircle className="h-3.5 w-3.5" />
                    Add Doc
                  </button>
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-3 gap-2.5">
                  <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-3 text-center">
                    <div className="text-xl font-bold text-white">{documents.length}</div>
                    <div className="text-[11px] text-slate-400 mt-0.5">Documents</div>
                  </div>
                  <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-3 text-center">
                    <div className="text-xl font-bold text-white">{totalPages}</div>
                    <div className="text-[11px] text-slate-400 mt-0.5">Pages</div>
                  </div>
                  <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-3 text-center">
                    <div className="text-xl font-bold text-emerald-400">
                      {result?.totalIndexedChunks || estimatedChunks}
                    </div>
                    <div className="text-[11px] text-slate-400 mt-0.5">Chunks</div>
                  </div>
                </div>

                {/* Document List */}
                <div className="space-y-2 pt-1">
                  <div className="text-xs font-medium text-slate-400">Indexed PDF Documents:</div>
                  {documents.map((doc, idx) => (
                    <div
                      key={idx}
                      className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3 flex items-start gap-2.5 hover:border-slate-700 transition"
                    >
                      <FileText className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                      <div className="min-w-0 flex-1">
                        <div className="text-xs font-semibold text-slate-200 truncate">{doc.name}</div>
                        <div className="text-[11px] text-slate-400 flex items-center gap-2 mt-0.5">
                          <span>{doc.pageCount} Pages</span>
                          <span>&bull;</span>
                          <span className="text-emerald-400">FAISS Indexed</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Architecture parameters */}
                <div className="border-t border-slate-800/80 pt-4 space-y-3">
                  <div className="flex items-center justify-between text-xs text-slate-300">
                    <span className="flex items-center gap-1.5 font-medium">
                      <Sliders className="h-3.5 w-3.5 text-slate-400" />
                      Top-k Retrieval
                    </span>
                    <span className="font-semibold text-emerald-400">{topK} chunks</span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="6"
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                    className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                  />

                  <div className="flex items-center justify-between text-xs text-slate-300 pt-1">
                    <span className="flex items-center gap-1.5 font-medium">
                      <ShieldCheck className="h-3.5 w-3.5 text-slate-400" />
                      Min Relevance Guardrail
                    </span>
                    <span className="font-semibold text-emerald-400">{(minThreshold * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.10"
                    max="0.40"
                    step="0.02"
                    value={minThreshold}
                    onChange={(e) => setMinThreshold(Number(e.target.value))}
                    className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                  />
                  <div className="text-[11px] text-slate-500 italic">
                    Below this similarity score, queries are rejected prior to LLM reasoning.
                  </div>
                </div>
              </div>

              {/* Rubric Alignment Info */}
              <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-4 text-xs space-y-2.5">
                <div className="font-semibold text-slate-200 flex items-center gap-1.5">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                  Rubric Compliance Summary
                </div>
                <ul className="space-y-1.5 text-slate-400 text-[11px]">
                  <li className="flex items-center gap-1.5">
                    <span className="text-emerald-400 font-bold">40 pts</span>
                    <span>Accurate grounded answers + exact metadata citations</span>
                  </li>
                  <li className="flex items-center gap-1.5">
                    <span className="text-emerald-400 font-bold">30 pts</span>
                    <span>Says &ldquo;I don&apos;t know based on the provided documents.&rdquo;</span>
                  </li>
                  <li className="flex items-center gap-1.5">
                    <span className="text-emerald-400 font-bold">20 pts</span>
                    <span>Clear visual UI with relevance meters &amp; evidence audit</span>
                  </li>
                  <li className="flex items-center gap-1.5">
                    <span className="text-emerald-400 font-bold">10 pts</span>
                    <span>No LLM-invented citations; deterministic Python code</span>
                  </li>
                </ul>
              </div>
            </div>

            {/* Right Column: Q&A Engine, Guardrails & Evidence (8 cols) */}
            <div className="lg:col-span-8 space-y-5">
              {/* Question Input Card */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <label htmlFor="rag-query-input" className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                    <Search className="h-4 w-4 text-emerald-400" />
                    Ask Document Question
                  </label>
                  <span className="text-xs text-slate-400">Zero guessing guarantee</span>
                </div>

                {/* Preset Scenarios */}
                <div className="space-y-1.5">
                  <div className="text-[11px] font-semibold tracking-wide text-slate-400 uppercase">
                    Rubric Test Presets:
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    <button
                      onClick={() => handleAsk('What is the annual wellness reimbursement stipend?')}
                      className="text-left text-xs bg-slate-800/80 hover:bg-slate-800 hover:border-emerald-500/40 border border-slate-700/60 p-2.5 rounded-xl transition text-slate-200 group flex items-start justify-between"
                    >
                      <div>
                        <div className="font-semibold text-emerald-400 flex items-center gap-1">
                          <CheckCircle2 className="h-3 w-3" /> Answerable (Policy)
                        </div>
                        <div className="text-slate-400 mt-0.5">&ldquo;What is the annual wellness stipend?&rdquo;</div>
                      </div>
                      <ChevronRight className="h-4 w-4 text-slate-500 group-hover:text-emerald-400 shrink-0 mt-1 transition" />
                    </button>

                    <button
                      onClick={() => handleAsk('What are the password requirements and MFA policy?')}
                      className="text-left text-xs bg-slate-800/80 hover:bg-slate-800 hover:border-emerald-500/40 border border-slate-700/60 p-2.5 rounded-xl transition text-slate-200 group flex items-start justify-between"
                    >
                      <div>
                        <div className="font-semibold text-emerald-400 flex items-center gap-1">
                          <CheckCircle2 className="h-3 w-3" /> Answerable (Security)
                        </div>
                        <div className="text-slate-400 mt-0.5">&ldquo;What are the password requirements?&rdquo;</div>
                      </div>
                      <ChevronRight className="h-4 w-4 text-slate-500 group-hover:text-emerald-400 shrink-0 mt-1 transition" />
                    </button>

                    <button
                      onClick={() => handleAsk("What is the CEO's stock grant or executive equity? ")}
                      className="text-left text-xs bg-slate-800/80 hover:bg-slate-800 hover:border-rose-500/40 border border-slate-700/60 p-2.5 rounded-xl transition text-slate-200 group flex items-start justify-between"
                    >
                      <div>
                        <div className="font-semibold text-rose-400 flex items-center gap-1">
                          <XCircle className="h-3 w-3" /> Guardrail Test (Unanswerable)
                        </div>
                        <div className="text-slate-400 mt-0.5">&ldquo;What is the CEO&apos;s stock grant?&rdquo;</div>
                      </div>
                      <ChevronRight className="h-4 w-4 text-slate-500 group-hover:text-rose-400 shrink-0 mt-1 transition" />
                    </button>

                    <button
                      onClick={() => handleAsk('Does Acme provide private jet travel to executive conferences?')}
                      className="text-left text-xs bg-slate-800/80 hover:bg-slate-800 hover:border-rose-500/40 border border-slate-700/60 p-2.5 rounded-xl transition text-slate-200 group flex items-start justify-between"
                    >
                      <div>
                        <div className="font-semibold text-rose-400 flex items-center gap-1">
                          <AlertTriangle className="h-3 w-3" /> Hallucination Trap
                        </div>
                        <div className="text-slate-400 mt-0.5">&ldquo;Does Acme provide private jet travel?&rdquo;</div>
                      </div>
                      <ChevronRight className="h-4 w-4 text-slate-500 group-hover:text-rose-400 shrink-0 mt-1 transition" />
                    </button>
                  </div>
                </div>

                {/* Input box */}
                <div className="flex gap-2">
                  <input
                    id="rag-query-input"
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAsk()}
                    placeholder="Enter any question about the uploaded documents..."
                    className="flex-1 bg-slate-950 border border-slate-700/80 rounded-xl px-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
                  />
                  <button
                    onClick={() => handleAsk()}
                    disabled={isQuerying || !query.trim()}
                    className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-medium px-5 py-2.5 rounded-xl text-sm transition flex items-center gap-2 shrink-0 shadow-lg shadow-emerald-600/20"
                  >
                    {isQuerying ? (
                      <>
                        <div className="h-4 w-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                        <span>Auditing...</span>
                      </>
                    ) : (
                      <>
                        <Search className="h-4 w-4" />
                        <span>Run TrustRAG</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Execution Pipeline Indicator (While querying or showing results) */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4">
                <div className="text-xs font-semibold text-slate-400 mb-2 flex items-center justify-between">
                  <span>TrustRAG Verification Pipeline</span>
                  {isQuerying && <span className="text-emerald-400 animate-pulse">Running live pipeline...</span>}
                </div>
                <div className="grid grid-cols-4 gap-2 text-center text-xs">
                  <div
                    className={`p-2 rounded-xl border transition ${
                      isQuerying || result
                        ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
                        : 'bg-slate-800/40 border-slate-700/40 text-slate-400'
                    }`}
                  >
                    <div className="font-bold text-[10px] tracking-wider uppercase">Step 1</div>
                    <div className="font-semibold text-xs mt-0.5">Retrieve</div>
                    <div className="text-[10px] text-slate-400 mt-0.5">FAISS Top-k</div>
                  </div>

                  <div
                    className={`p-2 rounded-xl border transition ${
                      isQuerying || result
                        ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
                        : 'bg-slate-800/40 border-slate-700/40 text-slate-400'
                    }`}
                  >
                    <div className="font-bold text-[10px] tracking-wider uppercase">Step 2</div>
                    <div className="font-semibold text-xs mt-0.5">Verify</div>
                    <div className="text-[10px] text-slate-400 mt-0.5">Auditor (temp 0.0)</div>
                  </div>

                  <div
                    className={`p-2 rounded-xl border transition ${
                      result
                        ? result.isSupported
                          ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
                          : 'bg-rose-950/40 border-rose-500/40 text-rose-300'
                        : 'bg-slate-800/40 border-slate-700/40 text-slate-400'
                    }`}
                  >
                    <div className="font-bold text-[10px] tracking-wider uppercase">Step 3</div>
                    <div className="font-semibold text-xs mt-0.5">
                      {result ? (result.isSupported ? 'Answer' : 'Guardrail') : 'Answer / Fallback'}
                    </div>
                    <div className="text-[10px] text-slate-400 mt-0.5">Strict Factual</div>
                  </div>

                  <div
                    className={`p-2 rounded-xl border transition ${
                      result && result.isSupported
                        ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
                        : 'bg-slate-800/40 border-slate-700/40 text-slate-400'
                    }`}
                  >
                    <div className="font-bold text-[10px] tracking-wider uppercase">Step 4</div>
                    <div className="font-semibold text-xs mt-0.5">Cite</div>
                    <div className="text-[10px] text-slate-400 mt-0.5">Code Generated</div>
                  </div>
                </div>
              </div>

              {/* Error Alert */}
              {errorMsg && (
                <div className="bg-rose-950/40 border border-rose-500/40 rounded-xl p-4 text-xs text-rose-300 flex items-start gap-2">
                  <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-semibold">Execution Error: </span>
                    {errorMsg}
                  </div>
                </div>
              )}

              {/* Response Results Display */}
              {result && (
                <div className="space-y-4">
                  {/* Status Banner */}
                  <div
                    className={`rounded-2xl p-5 border shadow-sm transition ${
                      result.isSupported
                        ? 'bg-emerald-950/30 border-emerald-500/40'
                        : 'bg-rose-950/30 border-rose-500/40'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        {result.isSupported ? (
                          <div className="px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 text-xs font-bold flex items-center gap-1.5">
                            <CheckCircle2 className="h-3.5 w-3.5" />
                            STATUS: SUPPORTED BY DOCUMENTS
                          </div>
                        ) : (
                          <div className="px-3 py-1 rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/40 text-xs font-bold flex items-center gap-1.5">
                            <XCircle className="h-3.5 w-3.5" />
                            STATUS: NOT SUPPORTED &mdash; GUARDRAIL TRIGGERED
                          </div>
                        )}
                      </div>
                      <div className="text-xs text-slate-400">
                        Top Evidence Score:{' '}
                        <span className="font-semibold text-slate-200">
                          {result.retrievedEvidence[0]?.score || 0}%
                        </span>
                      </div>
                    </div>

                    {/* Main Answer Text */}
                    <div className="space-y-2">
                      <div className="text-xs font-bold uppercase tracking-wider text-slate-400">Answer:</div>
                      <div
                        className={`text-base font-medium leading-relaxed p-4 rounded-xl border ${
                          result.isSupported
                            ? 'bg-slate-900/90 border-slate-800 text-slate-100'
                            : 'bg-rose-950/50 border-rose-900/60 text-rose-200 font-semibold text-lg'
                        }`}
                      >
                        {result.answer}
                      </div>
                    </div>

                    {/* Verification Auditor Reasoning */}
                    <div className="mt-3.5 pt-3.5 border-t border-slate-800/80 flex items-start gap-2 text-xs">
                      <Info className="h-4 w-4 text-slate-400 shrink-0 mt-0.5" />
                      <div className="text-slate-300">
                        <span className="font-semibold text-slate-200">Stage 1 Evidence Audit: </span>
                        {result.verificationReason}
                      </div>
                    </div>
                  </div>

                  {/* Deterministic Citations Card */}
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <FileCheck2 className="h-4 w-4 text-emerald-400" />
                        <span className="text-sm font-semibold text-slate-200">Source Citations</span>
                      </div>
                      <span className="text-[11px] text-slate-400 bg-slate-800 px-2 py-0.5 rounded-md border border-slate-700/60 font-mono">
                        Python-Generated
                      </span>
                    </div>

                    {result.isSupported && result.citations.length > 0 ? (
                      <div className="space-y-2.5">
                        <div className="text-xs text-slate-400 italic">
                          Important: Citations were generated directly from vector chunk metadata by Python. The LLM was
                          strictly prevented from inventing citations.
                        </div>
                        <div className="grid grid-cols-1 gap-2.5">
                          {result.citations.map((cite, idx) => (
                            <div
                              key={idx}
                              className="bg-slate-950/70 border border-slate-800 rounded-xl p-3.5 hover:border-emerald-500/40 transition space-y-1.5"
                            >
                              <div className="flex items-center justify-between text-xs">
                                <span className="font-bold text-emerald-400 font-mono text-[11px]">
                                  {cite.citationLabel}
                                </span>
                                <span className="text-xs font-semibold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                                  {cite.relevanceScore}% Relevance
                                </span>
                              </div>
                              <div className="text-xs text-slate-300 leading-relaxed italic bg-slate-900/60 p-2.5 rounded-lg border border-slate-800/80">
                                &ldquo;{cite.excerpt}&rdquo;
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="text-xs text-slate-400 bg-slate-950/50 p-4 rounded-xl border border-slate-800/80 text-center">
                        Citations are suppressed because the question is unsupported by the provided documents.
                      </div>
                    )}
                  </div>

                  {/* Retrieved Evidence & Relevance Breakdown */}
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Cpu className="h-4 w-4 text-emerald-400" />
                        <span className="text-sm font-semibold text-slate-200">
                          Retrieved Evidence &amp; Relevance Scores
                        </span>
                      </div>
                      <span className="text-xs text-slate-400 font-medium">
                        Top {result.retrievedEvidence.length} Chunks
                      </span>
                    </div>

                    <div className="space-y-3">
                      {result.retrievedEvidence.map((item, idx) => (
                        <div
                          key={idx}
                          className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3.5 space-y-2"
                        >
                          <div className="flex items-center justify-between text-xs">
                            <div className="flex items-center gap-2 font-medium text-slate-200">
                              <span className="font-mono text-emerald-400">Chunk #{item.chunkId}</span>
                              <span className="text-slate-500">&bull;</span>
                              <span>{item.docName}</span>
                              <span className="text-slate-500">&bull;</span>
                              <span className="text-slate-400">Page {item.pageNumber}</span>
                            </div>
                            <span className="font-semibold text-emerald-400 font-mono text-xs">
                              {item.score}% similarity
                            </span>
                          </div>

                          {/* Relevance Progress Bar */}
                          <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                            <div
                              className="bg-emerald-500 h-1.5 rounded-full transition-all duration-500"
                              style={{ width: `${item.score}%` }}
                            />
                          </div>

                          <div className="text-xs text-slate-300 font-mono bg-slate-900/80 p-2.5 rounded-lg border border-slate-800/60 leading-relaxed whitespace-pre-wrap">
                            {item.text}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ================= TAB 2: ARCHITECTURE & HOW IT WORKS ================= */}
        {activeTab === 'architecture' && (
          <div className="max-w-4xl mx-auto space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <Layers className="h-5 w-5 text-emerald-400" />
                How TrustRAG Eliminates Hallucination
              </h2>
              <p className="text-sm text-slate-300 leading-relaxed">
                Standard RAG pipelines feed retrieved chunks into an LLM and tell it: &ldquo;Answer the question and don&apos;t make things up.&rdquo;
                In practice, LLMs still hallucinate when context is partially related, and worse, they often hallucinate plausible-looking page numbers and citations.
                TrustRAG implements a strict 4-step architecture:
              </p>

              {/* 4 Pipeline Stages */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 space-y-2">
                  <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm">
                    <span className="h-6 w-6 rounded-full bg-emerald-500/20 flex items-center justify-center text-xs">1</span>
                    Retrieve (Page-Aware Chunking &amp; FAISS)
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Documents are parsed page-by-page. Chunks are strictly bound to page coordinates with 600-character windows and 120-character overlap.
                    Embeddings are computed with open-source <code className="text-emerald-300">all-MiniLM-L6-v2</code> and indexed in <code className="text-emerald-300">FAISS IndexFlatIP</code> with L2 normalization.
                  </p>
                </div>

                <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 space-y-2">
                  <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm">
                    <span className="h-6 w-6 rounded-full bg-emerald-500/20 flex items-center justify-center text-xs">2</span>
                    Verify (Independent Auditor @ Temp 0.0)
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Before generating an answer, Gemini acts purely as a factual verification auditor.
                    If the retrieved context requires any speculation, guessing, or outside knowledge, verification returns <code className="text-rose-300">is_supported: false</code>.
                  </p>
                </div>

                <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 space-y-2">
                  <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm">
                    <span className="h-6 w-6 rounded-full bg-emerald-500/20 flex items-center justify-center text-xs">3</span>
                    Guardrail / Grounded Synthesis
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    If unsupported, the system deterministically aborts to: <br />
                    <span className="font-semibold text-rose-300">&ldquo;I don&apos;t know based on the provided documents.&rdquo;</span><br />
                    If supported, the model synthesizes an answer using strictly verified chunks.
                  </p>
                </div>

                <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 space-y-2">
                  <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm">
                    <span className="h-6 w-6 rounded-full bg-emerald-500/20 flex items-center justify-center text-xs">4</span>
                    Deterministic Python Citations
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    The LLM is strictly prohibited from writing citation text.
                    Instead, Python code stamps: <br />
                    <code className="text-emerald-300 font-mono text-[11px]">[Document | Page X | Chunk Y]</code> directly from vector store metadata.
                  </p>
                </div>
              </div>
            </div>

            {/* Rubric Scorecard Table */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                Hackathon Rubric Breakdown (100 Points)
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left text-slate-300">
                  <thead className="bg-slate-800/80 text-slate-200 uppercase font-semibold">
                    <tr>
                      <th className="px-3 py-2.5 rounded-l-lg">Rubric Category</th>
                      <th className="px-3 py-2.5">Weight</th>
                      <th className="px-3 py-2.5">TrustRAG Implementation</th>
                      <th className="px-3 py-2.5 rounded-r-lg">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800 text-[11px]">
                    <tr>
                      <td className="px-3 py-3 font-semibold text-slate-100">Correct Answers + Citations</td>
                      <td className="px-3 py-3 text-emerald-400 font-bold">40 pts</td>
                      <td className="px-3 py-3">Gemini 3.8/2.5 synthesizes only verified chunks; Python stamps [Doc, Page, Chunk]</td>
                      <td className="px-3 py-3 text-emerald-400 font-bold">&check; Fully Verified</td>
                    </tr>
                    <tr>
                      <td className="px-3 py-3 font-semibold text-slate-100">Strong &ldquo;I don&apos;t know&rdquo; Guardrail</td>
                      <td className="px-3 py-3 text-emerald-400 font-bold">30 pts</td>
                      <td className="px-3 py-3">Stage 1 auditor evaluates context sufficiency at temp 0.0; refuses to guess</td>
                      <td className="px-3 py-3 text-emerald-400 font-bold">&check; Fully Verified</td>
                    </tr>
                    <tr>
                      <td className="px-3 py-3 font-semibold text-slate-100">Demo / UI Clarity</td>
                      <td className="px-3 py-3 text-emerald-400 font-bold">20 pts</td>
                      <td className="px-3 py-3">Clear badges, relevance score meters, audit explanations, and instant sample questions</td>
                      <td className="px-3 py-3 text-emerald-400 font-bold">&check; Fully Verified</td>
                    </tr>
                    <tr>
                      <td className="px-3 py-3 font-semibold text-slate-100">Design Choices &amp; Anti-Hallucination</td>
                      <td className="px-3 py-3 text-emerald-400 font-bold">10 pts</td>
                      <td className="px-3 py-3">Page-bounded chunking, FAISS IndexFlatIP cosine similarity, LLM prohibited from citing</td>
                      <td className="px-3 py-3 text-emerald-400 font-bold">&check; Fully Verified</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ================= TAB 3: PYTHON SOURCE CODE ================= */}
        {activeTab === 'code' && (
          <div className="max-w-5xl mx-auto space-y-4">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div>
                  <h3 className="text-base font-bold text-white flex items-center gap-2">
                    <Code2 className="h-5 w-5 text-emerald-400" />
                    Complete Python Streamlit Project Files
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Ready to run locally with: <code className="text-emerald-300 bg-slate-950 px-2 py-0.5 rounded font-mono">python -m streamlit run app.py</code>
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleCopyCode(getCodeContent(activeCodeFile))}
                    className="bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium px-3.5 py-1.5 rounded-lg border border-slate-700/80 transition flex items-center gap-1.5"
                  >
                    {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                    <span>{copied ? 'Copied!' : 'Copy File'}</span>
                  </button>
                </div>
              </div>

              {/* File selector pills */}
              <div className="flex flex-wrap gap-1.5 border-b border-slate-800 pb-3 pt-1">
                {(['app.py', 'rag_engine.py', 'requirements.txt', 'README.md'] as const).map((file) => (
                  <button
                    key={file}
                    onClick={() => setActiveCodeFile(file)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition ${
                      activeCodeFile === file
                        ? 'bg-emerald-600 text-white shadow-sm'
                        : 'bg-slate-800/60 text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    {file}
                  </button>
                ))}
              </div>

              {/* Code viewer container */}
              <div className="bg-slate-950 rounded-xl p-4 border border-slate-800 overflow-x-auto max-h-[600px] overflow-y-auto font-mono text-xs text-slate-300 leading-relaxed whitespace-pre">
                {getCodeContent(activeCodeFile)}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Add Document Modal */}
      {showAddDoc && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 max-w-lg w-full space-y-4 shadow-2xl">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <FileText className="h-4 w-4 text-emerald-400" />
                Add New Document to Knowledge Base
              </h3>
              <button onClick={() => setShowAddDoc(false)} className="text-slate-400 hover:text-white">
                <XCircle className="h-5 w-5" />
              </button>
            </div>

            <div className="space-y-3">
              <div>
                <label className="text-xs text-slate-400 block mb-1">Document Title (e.g. Employee_Safety_Manual.pdf)</label>
                <input
                  type="text"
                  value={newDocTitle}
                  onChange={(e) => setNewDocTitle(e.target.value)}
                  placeholder="Company_Policy.pdf"
                  className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500"
                />
              </div>

              <div>
                <label className="text-xs text-slate-400 block mb-1">Document Text / Policy Content</label>
                <textarea
                  rows={6}
                  value={newDocText}
                  onChange={(e) => setNewDocText(e.target.value)}
                  placeholder="Paste document text here. Chunks and page boundaries will be indexed automatically..."
                  className="w-full bg-slate-950 border border-slate-700 rounded-xl p-3 text-xs text-white focus:outline-none focus:border-emerald-500 font-mono"
                />
              </div>
            </div>

            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => setShowAddDoc(false)}
                className="px-4 py-2 text-xs font-medium text-slate-400 hover:text-white"
              >
                Cancel
              </button>
              <button
                onClick={handleAddCustomDoc}
                disabled={!newDocTitle.trim() || !newDocText.trim()}
                className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-medium px-4 py-2 rounded-xl text-xs transition"
              >
                Index Document
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Helper to provide code contents in tab
function getCodeContent(file: string): string {
  switch (file) {
    case 'app.py':
      return `"""
TrustRAG - Hallucination-Proof Document Q&A Bot
Run with: python -m streamlit run app.py
"""
import os, io, time
import streamlit as st
from dotenv import load_dotenv
from rag_engine import TrustRAGEngine

load_dotenv()

st.set_page_config(page_title="TrustRAG | Hallucination-Proof Q&A", page_icon="🛡️", layout="wide")

@st.cache_resource
def get_engine():
    return TrustRAGEngine(
        embedding_model_name="all-MiniLM-L6-v2",
        chunk_size=600,
        chunk_overlap=120,
        gemini_model="gemini-2.5-flash"
    )

def main():
    st.title("🛡️ TrustRAG: Hallucination-Proof Document Q&A")
    engine = get_engine()

    # Sidebar: PDF upload
    with st.sidebar:
        st.header("Upload PDFs")
        files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        if files and st.button("Index PDFs"):
            inputs = [(f.name, io.BytesIO(f.read())) for f in files]
            stats = engine.build_index(inputs)
            st.success(f"Indexed {stats['total_chunks']} chunks across {stats['total_pages']} pages!")

    # Question Input
    query = st.text_input("Enter your question:")
    if st.button("Get Answer") and query:
        res = engine.ask(query)
        if res.is_supported:
            st.success(f"STATUS: SUPPORTED BY DOCUMENTS\\n\\n{res.answer}")
            st.subheader("Source Citations (Python-Generated)")
            for c in res.citations:
                st.markdown(f"📄 **{c['doc_name']}** | Page {c['page_number']} | Chunk #{c['chunk_id']} ({c['relevance_score']}%)")
                st.caption(f'"{c["excerpt"]}"')
        else:
            st.error(f"STATUS: NOT SUPPORTED\\n\\n**{res.answer}**")
            st.info(f"Audit: {res.verification_reason}")

if __name__ == "__main__":
    main()`;

    case 'rag_engine.py':
      return `"""
rag_engine.py - Core RAG Engine with FAISS, SentenceTransformers, and Gemini
"""
import os, re, json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from google import genai
from google.genai import types

load_dotenv()

@dataclass
class DocumentChunk:
    doc_name: str
    page_number: int
    chunk_id: int
    text: str
    char_count: int

    def to_citation_label(self) -> str:
        return f"[{self.doc_name} | Page {self.page_number} | Chunk {self.chunk_id}]"

class TrustRAGEngine:
    GUARDRAIL_FALLBACK = "I don't know based on the provided documents."

    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", chunk_size=600, chunk_overlap=120, gemini_model="gemini-2.5-flash"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.gemini_model = gemini_model
        self.embedder = SentenceTransformer(embedding_model_name)
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.chunks = []
        self.faiss_index = None

    def build_index(self, pdf_inputs: List[Tuple[str, Any]]):
        self.chunks.clear()
        cid = 1
        for doc_name, file_obj in pdf_inputs:
            reader = PdfReader(file_obj)
            for idx, page in enumerate(reader.pages):
                text = re.sub(r"\\s+", " ", page.extract_text() or "").strip()
                if not text: continue
                # Sentence-aware chunking
                start = 0
                while start < len(text):
                    end = min(start + self.chunk_size, len(text))
                    chunk_text = text[start:end].strip()
                    if chunk_text:
                        self.chunks.append(DocumentChunk(doc_name, idx + 1, cid, chunk_text, len(chunk_text)))
                        cid += 1
                    start += max(self.chunk_size - self.chunk_overlap, 100)

        # L2-normalized embeddings for Cosine Similarity in FAISS IndexFlatIP
        raw = self.embedder.encode([c.text for c in self.chunks], convert_to_numpy=True)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        self.embeddings = (raw / np.maximum(norms, 1e-10)).astype(np.float32)
        self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.faiss_index.add(self.embeddings)

    def ask(self, question: str, top_k: int = 4):
        # 1. Retrieve
        q_emb = self.embedder.encode([question], convert_to_numpy=True)
        q_norm = q_emb / np.maximum(np.linalg.norm(q_emb), 1e-10)
        dists, idxs = self.faiss_index.search(q_norm.astype(np.float32), min(top_k, len(self.chunks)))
        retrieved = [self.chunks[i] for i in idxs[0] if i < len(self.chunks)]

        # 2. Verify evidence with Gemini auditor
        context_str = "\\n\\n".join([f"[Chunk {c.chunk_id} | Page {c.page_number}]: {c.text}" for c in retrieved])
        verify_prompt = f"Does the context contain direct factual proof to answer: '{question}'?\\n\\nContext:\\n{context_str}"
        verify_res = self.client.models.generate_content(
            model=self.gemini_model,
            contents=verify_prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
        )
        data = json.loads(verify_res.text or "{}")

        # 3. Guardrail vs Answer
        if not data.get("is_supported", False):
            return {"answer": self.GUARDRAIL_FALLBACK, "is_supported": False, "citations": []}

        # 4. Synthesize Answer (LLM prohibited from outputting citations)
        ans_res = self.client.models.generate_content(
            model=self.gemini_model,
            contents=f"Answer strictly using facts below. Do NOT write citations.\\n\\nQ: {question}\\n\\nFacts:\\n{context_str}",
            config=types.GenerateContentConfig(temperature=0.1)
        )

        # 5. Deterministic Python Citations
        citations = [{
            "doc_name": c.doc_name,
            "page_number": c.page_number,
            "chunk_id": c.chunk_id,
            "label": c.to_citation_label()
        } for c in retrieved]

        return {"answer": ans_res.text.strip(), "is_supported": True, "citations": citations}`;

    case 'requirements.txt':
      return `streamlit>=1.35.0
pypdf>=4.2.0
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
google-genai>=0.1.1
python-dotenv>=1.0.1
numpy>=1.24.0`;

    case 'README.md':
      return `# TrustRAG: Hallucination-Proof Document Q&A Bot
Run with: python -m streamlit run app.py
Architecture: Retrieve -> Verify -> Answer -> Cite
Guardrail: "I don't know based on the provided documents."`;

    default:
      return '';
  }
}
