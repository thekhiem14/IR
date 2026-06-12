"""
Student Server v4 - đa năng (luật + tài liệu thường)
=====================================================
Sửa các vấn đề thấy từ log:
  - 2 câu sai (5 và 7) đều do RETRIEVE không lấy được chunk chứa đáp án
  - Chunking trước cắt ngang danh sách (Học kỳ 1, 2, 3...)
  - Rerank "Điều X" vô tác dụng vì doc thực tế không phải luật

Cải tiến chính:
  1. Auto-detect doc là luật hay tài liệu thường (nếu thấy nhiều "Điều X" -> luật)
  2. Chunking nhỏ hơn (400 char, overlap 150) -> liệt kê không bị cắt ngang
  3. Multi-query retrieval: query bằng (stem) + (mỗi choice) + (stem+choices)
     -> union -> coverage cao hơn nhiều
  4. TOP_K_FINAL = 12 chunks -> LLM có nhiều ngữ cảnh hơn
  5. Bỏ rerank luật-specific, dùng rerank by max-score (chunk khớp ít nhất 1 query)
  6. Log thêm thông tin debug
"""

import os
import re
import io
import json
import uuid
import time
import pickle
import logging
from collections import Counter, defaultdict
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from rank_bm25 import BM25Okapi

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# =============================================================
# CONFIG
# =============================================================
STUDENT_ID    = os.getenv("STUDENT_ID", "B22DCDT171")
TEACHER_PROXY = os.getenv("TEACHER_PROXY",
                          "http://192.168.50.218:8000/api/v1/proxy")
LLM_MODEL     = "gpt-4o-mini"
EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", "./models/vietnamese-sbert")

CHUNK_SIZE    = 400      # nhỏ hơn để liệt kê không bị cắt ngang
CHUNK_OVERLAP = 150      # overlap LỚN (~37%) để giữ ngữ cảnh kép
TOP_K_PER_QUERY = 8      # mỗi sub-query lấy bao nhiêu
TOP_K_FINAL    = 12      # ngữ cảnh cuối gửi LLM
ALPHA_SEMANTIC = 0.55
MAX_CTX_CHARS  = 7000    # tăng budget context
LLM_TIMEOUT_S  = 20

VOTE_TEMPS = [0.0, 0.0, 0.3, 0.5, 0.7]

LOG_PATH = "ask_log.jsonl"
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./storage/vectordb.pkl")
VECTOR_DB_VERSION = 1

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("student-server")

# =============================================================
# SCHEMAS
# =============================================================
class UploadRequest(BaseModel):
    doc_id: Optional[str] = None
    text: str

class UploadResponse(BaseModel):
    status: str
    doc_id: Optional[str] = None
    chunks: int

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: List[str] = []

# =============================================================
# GLOBAL STATE
# =============================================================
class RAGStore:
    def __init__(self):
        self.embedder: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[str] = []
        self.bm25: Optional[BM25Okapi] = None
        self.doc_id: Optional[str] = None
        self.is_law_doc: bool = False
        self.dim: int = 0
        self.ask_counter: int = 0

    def load_embedder(self):
        if self.embedder is not None:
            return
        if not os.path.isdir(EMBED_MODEL_PATH):
            raise RuntimeError(f"Không thấy model tại {EMBED_MODEL_PATH}")
        log.info("Loading embedder from local: %s", EMBED_MODEL_PATH)
        self.embedder = SentenceTransformer(EMBED_MODEL_PATH)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        log.info("Embedder ready. dim=%d", self.dim)

store = RAGStore()
llm_client = OpenAI(base_url=TEACHER_PROXY, api_key=STUDENT_ID,
                    timeout=LLM_TIMEOUT_S)

# =============================================================
# AUTO-DETECT doc type
# =============================================================
LAW_PATTERN = re.compile(r"Điều\s+\d+[a-z]?\s*[.:]?", re.IGNORECASE)

def is_law_document(text: str) -> bool:
    """Đếm số 'Điều X' - nếu >= 5 thì là văn bản luật."""
    matches = LAW_PATTERN.findall(text)
    return len(matches) >= 5

# =============================================================
# CHUNKING
# =============================================================
LAW_BOUNDARY = re.compile(
    r"(?:^|\n)\s*("
    r"Điều\s+\d+[a-z]?\s*[.:]?"
    r"|Chương\s+[IVXLCDM\d]+"
    r"|Mục\s+\d+"
    r"|Phần\s+(?:thứ\s+)?[IVXLCDM\d]+"
    r")",
    re.IGNORECASE
)

def smart_chunk_law(text: str, size: int, overlap: int) -> List[str]:
    """Cho văn bản luật - split theo Điều."""
    text = re.sub(r"[ \t]+", " ", text)
    parts = LAW_BOUNDARY.split(text)
    segments: List[str] = []
    if len(parts) > 1:
        head = parts[0].strip()
        if head:
            segments.append(head)
        for i in range(1, len(parts), 2):
            marker = parts[i].strip()
            body   = parts[i+1].strip() if i+1 < len(parts) else ""
            segments.append(f"{marker} {body}".strip())
    else:
        segments = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    return _apply_size_budget(segments, size, overlap)

def smart_chunk_general(text: str, size: int, overlap: int) -> List[str]:
    """
    Cho tài liệu thường (PTIT, văn bản hành chính, ...).
    Split mềm theo dòng + paragraph, KHÔNG ép theo cấu trúc cứng.
    Quan trọng: overlap LỚN để các liệt kê (HK1, HK2, HK3...) luôn xuất hiện
    cùng nhau ở ít nhất 1 chunk.
    """
    text = re.sub(r"[ \t]+", " ", text)
    # Split theo dòng trống (paragraph) hoặc xuống dòng đơn nếu dòng đó ngắn
    segments = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not segments:
        segments = [text.strip()]
    return _apply_size_budget(segments, size, overlap)

def _apply_size_budget(segments: List[str], size: int, overlap: int) -> List[str]:
    """Gom segment vào chunk theo budget, dùng sliding window cho segment quá dài."""
    chunks: List[str] = []
    buf = ""
    for seg in segments:
        if len(seg) > size:
            if buf:
                chunks.append(buf.strip()); buf = ""
            i = 0
            while i < len(seg):
                chunks.append(seg[i : i + size].strip())
                i += max(1, size - overlap)
            continue
        if not buf:
            buf = seg
        elif len(buf) + len(seg) + 1 <= size:
            buf = buf + "\n" + seg
        else:
            chunks.append(buf.strip())
            tail = buf[-overlap:] if len(buf) > overlap else ""
            buf = (tail + "\n" + seg).strip()
    if buf:
        chunks.append(buf.strip())
    return [c for c in chunks if len(c) >= 20]

# =============================================================
# TOKENIZE cho BM25
# =============================================================
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

def tokenize_vi(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]

# =============================================================
# EMBED + INDEX
# =============================================================
def embed_texts(texts: List[str]) -> np.ndarray:
    vecs = store.embedder.encode(
        texts, batch_size=32, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    return vecs.astype("float32")

def build_indices(chunks: List[str]):
    vecs = embed_texts(chunks)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    bm25 = BM25Okapi([tokenize_vi(c) for c in chunks])
    return index, bm25

# =============================================================
# PERSIST VECTOR DB
# =============================================================
def save_vector_db():
    if store.index is None or not store.chunks:
        return
    os.makedirs(os.path.dirname(VECTOR_DB_PATH) or ".", exist_ok=True)
    buf = io.BytesIO()
    faiss.write_index(store.index, faiss.PyCallbackIOWriter(buf.write))
    payload = {
        "version": VECTOR_DB_VERSION,
        "chunks": store.chunks,
        "doc_id": store.doc_id,
        "is_law_doc": store.is_law_doc,
        "faiss_bytes": buf.getvalue(),
    }
    tmp = VECTOR_DB_PATH + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, VECTOR_DB_PATH)
    log.info("VectorDB saved: %s (%d chunks)", VECTOR_DB_PATH, len(store.chunks))

def load_vector_db() -> bool:
    if not os.path.isfile(VECTOR_DB_PATH):
        return False
    try:
        with open(VECTOR_DB_PATH, "rb") as f:
            payload = pickle.load(f)
        if payload.get("version") != VECTOR_DB_VERSION:
            log.warning("VectorDB version mismatch, skip load")
            return False
        chunks = payload["chunks"]
        buf = io.BytesIO(payload["faiss_bytes"])
        index = faiss.read_index(faiss.PyCallbackIOReader(buf.read))
        bm25 = BM25Okapi([tokenize_vi(c) for c in chunks])
        store.chunks = chunks
        store.index = index
        store.bm25 = bm25
        store.doc_id = payload.get("doc_id")
        store.is_law_doc = payload.get("is_law_doc", False)
        log.info("VectorDB loaded: %s (%d chunks)", VECTOR_DB_PATH, len(chunks))
        return True
    except Exception as e:
        log.warning("Failed to load VectorDB: %s", e)
        return False

# =============================================================
# TÁCH STEM + 4 CHOICES
# =============================================================
CHOICE_SPLIT = re.compile(r"\s*([ABCD])\s*[.):\]]\s*", re.IGNORECASE)

def split_question(question: str) -> Tuple[str, Dict[str, str]]:
    parts = CHOICE_SPLIT.split(question)
    if len(parts) < 9:
        return question.strip(), {}
    stem = parts[0].strip()
    choices = {}
    for i in range(1, len(parts) - 1, 2):
        letter = parts[i].upper()
        choices[letter] = parts[i + 1].strip()
    if set(choices.keys()) >= {"A", "B", "C", "D"}:
        return stem, choices
    return question.strip(), {}

# =============================================================
# HYBRID RETRIEVE 1 QUERY
# =============================================================
def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)

def hybrid_retrieve_one(query: str, k: int, alpha: float
                        ) -> List[Tuple[int, float]]:
    if store.index is None or not store.chunks:
        return []
    n = len(store.chunks)
    k = min(k, n)

    q_vec = embed_texts([query])
    sem_scores, sem_idxs = store.index.search(q_vec, n)
    sem_arr = np.zeros(n, dtype="float32")
    for s, i in zip(sem_scores[0], sem_idxs[0]):
        if 0 <= i < n:
            sem_arr[i] = s

    bm25_scores = np.array(store.bm25.get_scores(tokenize_vi(query)),
                           dtype="float32")

    sem_n  = _minmax_norm(sem_arr)
    bm25_n = _minmax_norm(bm25_scores)
    hybrid = alpha * sem_n + (1.0 - alpha) * bm25_n

    top_idx = np.argsort(-hybrid)[:k]
    return [(int(i), float(hybrid[i])) for i in top_idx]

# =============================================================
# MULTI-QUERY RETRIEVAL (kernel của v4)
# =============================================================
_NUM_RE = re.compile(r"\d+")

# Các pattern "anchor" - nếu câu hỏi có những token này thì chunk chứa CHÍNH XÁC
# token đó phải được ưu tiên TỐI ĐA (vì không có nhiều chunk khác cũng có)
ANCHOR_PATTERNS = [
    re.compile(r"\bLO\s*\d+\b",                      re.IGNORECASE),  # LO1, LO 2
    re.compile(r"\bPI\s*\d+\.\d+\b",                 re.IGNORECASE),  # PI 1.1
    re.compile(r"\bHọc\s*kỳ\s*\d+\b",                re.IGNORECASE),  # Học kỳ 3
    re.compile(r"\b\d{7}\b"),                                          # mã ngành 7 số
    re.compile(r"\b(?:Điều|Khoản|Chương|Mục|Điểm)\s*\d+\b", re.IGNORECASE),
]

def extract_anchors(text: str) -> List[str]:
    """Trích các 'anchor' đặc biệt từ câu hỏi (LO2, Học kỳ 6, mã 7480107, ...)."""
    anchors = []
    for pat in ANCHOR_PATTERNS:
        for m in pat.findall(text):
            # Normalize: bỏ khoảng trắng dư
            normalized = re.sub(r"\s+", "", m).lower()
            if normalized not in anchors:
                anchors.append(normalized)
    return anchors

def chunks_containing(anchor: str) -> List[int]:
    """Quét toàn bộ chunks (KHÔNG qua embed/BM25) tìm chunk chứa anchor."""
    target = anchor.lower()
    out = []
    for i, c in enumerate(store.chunks):
        # So sánh sau khi bỏ khoảng trắng để bắt 'LO2' trong 'LO 2', 'lo2:', ...
        c_norm = re.sub(r"\s+", "", c).lower()
        if target in c_norm:
            out.append(i)
    return out

def multi_query_retrieve(stem: str,
                         choices: Dict[str, str],
                         full_question: str) -> List[int]:
    """
    Chiến lược 3 lớp:
      A) ANCHOR LOOKUP - quét cứng chunks chứa LO2, PI x.x, Học kỳ X, mã ngành
         -> bonus điểm RẤT CAO (chắc chắn đúng chunk cần tìm)
      B) MULTI-QUERY hybrid: stem + mỗi choice + full
         -> coverage rộng
      C) Bonus nhỏ cho chunks chứa các con số trùng câu hỏi
    """
    best_score: Dict[int, float] = defaultdict(float)

    # ========== A) Anchor lookup (PHẦN MỚI, mạnh nhất) ==========
    anchors = extract_anchors(full_question)
    anchor_hits: Dict[str, List[int]] = {}
    for a in anchors:
        hit_idxs = chunks_containing(a)
        anchor_hits[a] = hit_idxs
        for idx in hit_idxs:
            best_score[idx] += 1.0     # boost cứng, vượt mọi score hybrid

    # ========== B) Multi-query hybrid ==========
    queries: List[Tuple[str, float]] = []
    queries.append((stem if stem else full_question, 1.2))
    if choices:
        for letter in "ABCD":
            ch = choices.get(letter, "")
            if ch and len(ch) > 2:
                queries.append((f"{stem} {ch}" if stem else ch, 1.0))
    queries.append((full_question, 0.9))

    for q, w in queries:
        results = hybrid_retrieve_one(q, k=TOP_K_PER_QUERY, alpha=ALPHA_SEMANTIC)
        for idx, score in results:
            weighted = score * w
            # Cộng dồn thay vì max -> chunk được nhiều query khớp = quan trọng hơn
            best_score[idx] += weighted * 0.3

    # ========== C) Bonus số trùng (số >= 3 ký tự để tránh 1, 2, 3 nhiễu) ==========
    q_nums = [n for n in _NUM_RE.findall(full_question) if len(n) >= 3]
    if q_nums:
        for idx in list(best_score.keys()):
            chunk_low = store.chunks[idx].lower()
            for n in q_nums:
                if n in chunk_low:
                    best_score[idx] += 0.1

    ranked = sorted(best_score.items(), key=lambda x: -x[1])
    top = [idx for idx, _ in ranked[:TOP_K_FINAL]]

    # Log anchor hits để debug
    if anchors:
        log.info("Anchors=%s | hits=%s",
                 anchors, {a: len(idxs) for a, idxs in anchor_hits.items()})

    return top

# =============================================================
# PROMPT
# =============================================================
SYSTEM_PROMPT = (
    "Bạn là trợ lý trả lời câu hỏi trắc nghiệm dựa CHẶT CHẼ vào ngữ cảnh "
    "được trích từ tài liệu.\n"
    "QUY TẮC TUYỆT ĐỐI:\n"
    "1. Chỉ trả lời bằng ĐÚNG MỘT KÝ TỰ: A, B, C, hoặc D. Không thêm gì khác.\n"
    "2. Đọc kỹ câu hỏi:\n"
    "   - Có từ 'KHÔNG', 'KHÔNG PHẢI', 'SAI', 'TRỪ', 'NGOẠI TRỪ' "
    "-> tìm phương án TRÁI với ngữ cảnh.\n"
    "   - Có số (mã ngành, học kỳ, năm, tiền) -> tìm số TRÙNG KHỚP trong ngữ cảnh.\n"
    "   - Có tên riêng (LO1, LO2, PI 1.1, Điều X) -> tìm CHÍNH XÁC tên đó.\n"
    "3. So sánh từng phương án A/B/C/D với ngữ cảnh, loại trừ phương án trái, "
    "chọn phương án khớp nhất.\n"
    "4. Nếu ngữ cảnh không nói rõ, suy luận theo logic chung (không bỏ trống)."
)

USER_TEMPLATE = (
    "NGỮ CẢNH:\n{context}\n\n"
    "CÂU HỎI TRẮC NGHIỆM:\n{question}\n\n"
    "Đối chiếu từng phương án với ngữ cảnh. Trả lời DUY NHẤT 1 ký tự A/B/C/D:"
)

def build_context(chunks: List[str], budget: int = MAX_CTX_CHARS) -> str:
    out, used = [], 0
    for i, c in enumerate(chunks, 1):
        piece = f"[{i}] {c}"
        if used + len(piece) > budget:
            break
        out.append(piece); used += len(piece)
    return "\n\n".join(out) if out else "(không có ngữ cảnh)"

_ANSWER_RE = re.compile(r"\b([ABCD])\b")

def parse_answer(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().upper()
    if s and s[0] in "ABCD":
        return s[0]
    m = _ANSWER_RE.search(s)
    return m.group(1) if m else None

# =============================================================
# LLM + VOTING
# =============================================================
def _call_llm_once(question: str, context: str,
                   temperature: float) -> Optional[str]:
    try:
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(
                    context=context, question=question)},
            ],
            temperature=temperature,
            max_tokens=4,
        )
        return parse_answer(resp.choices[0].message.content or "")
    except Exception as e:
        log.warning("LLM call failed (temp=%s): %s", temperature, e)
        return None

def answer_with_voting(question: str, context: str,
                       deadline: float) -> Tuple[str, List[Optional[str]]]:
    votes: List[Optional[str]] = []
    for t in VOTE_TEMPS:
        if time.time() > deadline:
            log.warning("Vote deadline hit")
            break
        v = _call_llm_once(question, context, temperature=t)
        votes.append(v)
        valid = [x for x in votes if x]
        if valid:
            c = Counter(valid)
            top, n_top = c.most_common(1)[0]
            if n_top >= 3:
                log.info("Vote early-stop @ %d calls: %s -> %s",
                         len(votes), dict(c), top)
                return top, votes

    valid = [v for v in votes if v]
    if not valid:
        return "A", votes
    c = Counter(valid)
    top, n_top = c.most_common(1)[0]
    tied = [k for k, n in c.items() if n == n_top]
    if len(tied) > 1:
        for v in valid:
            if v in tied:
                return v, votes
    return top, votes

# =============================================================
# LOG
# =============================================================
def log_qa(payload: Dict[str, Any]):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("Failed to log Q&A: %s", e)

# =============================================================
# FASTAPI
# =============================================================
app = FastAPI(title="Student RAG Server v4 (multi-query)", version="4.0")

@app.on_event("startup")
def _startup():
    store.load_embedder()
    load_vector_db()
    log.info("Server ready. STUDENT_ID=%s", STUDENT_ID)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "student_id": STUDENT_ID,
        "has_index": store.index is not None,
        "n_chunks": len(store.chunks),
        "is_law_doc": store.is_law_doc,
        "ask_counter": store.ask_counter,
    }

@app.post("/upload", response_model=UploadResponse)
def upload(req: UploadRequest):
    try:
        if not req.text or not req.text.strip():
            raise HTTPException(status_code=400, detail="text rỗng")

        # Rotate log
        try:
            if os.path.exists(LOG_PATH):
                os.rename(LOG_PATH, f"{LOG_PATH}.{int(time.time())}.bak")
        except Exception:
            pass
        store.ask_counter = 0

        # Auto-detect doc type
        store.is_law_doc = is_law_document(req.text)
        log.info("UPLOAD: %d chars | is_law_doc=%s",
                 len(req.text), store.is_law_doc)

        if store.is_law_doc:
            chunks = smart_chunk_law(req.text, CHUNK_SIZE, CHUNK_OVERLAP)
        else:
            chunks = smart_chunk_general(req.text, CHUNK_SIZE, CHUNK_OVERLAP)

        if not chunks:
            raise HTTPException(status_code=400, detail="Không tách được chunk")

        log.info("UPLOAD: -> %d chunks", len(chunks))
        index, bm25 = build_indices(chunks)
        store.chunks = chunks
        store.index  = index
        store.bm25   = bm25
        store.doc_id = req.doc_id or f"doc_{uuid.uuid4().hex[:8]}"

        log_qa({"type": "upload",
                "doc_id": store.doc_id,
                "n_chars": len(req.text),
                "n_chunks": len(chunks),
                "is_law_doc": store.is_law_doc,
                "sample_chunks": chunks[:3]})

        try:
            save_vector_db()
        except Exception as e:
            log.warning("save_vector_db failed: %s", e)

        return UploadResponse(status="success",
                              doc_id=store.doc_id,
                              chunks=len(chunks))
    except HTTPException:
        raise
    except Exception as e:
        log.exception("UPLOAD error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    started = time.time()
    deadline = started + 55

    store.ask_counter += 1
    qid = store.ask_counter
    question = (req.question or "").strip()
    if not question:
        log_qa({"type": "ask", "qid": qid, "error": "empty", "answer": "A"})
        return AskResponse(answer="A", sources=[])

    sources: List[str] = []
    votes: List[Optional[str]] = []
    try:
        stem, choices = split_question(question)
        top_idxs = multi_query_retrieve(stem, choices, question)
        sources = [store.chunks[i] for i in top_idxs]
        context = build_context(sources)

        answer, votes = answer_with_voting(question, context, deadline)

        elapsed = time.time() - started
        log.info("ASK q%d -> %s | votes=%s | n_src=%d | %.1fs",
                 qid, answer, votes, len(sources), elapsed)

        log_qa({
            "type": "ask",
            "qid": qid,
            "question": question,
            "stem": stem,
            "choices": choices,
            "top_chunk_idxs": top_idxs,
            "sources_preview": [s[:200] for s in sources],
            "votes": votes,
            "answer": answer,
            "elapsed_s": round(elapsed, 2),
        })

        return AskResponse(answer=answer, sources=sources)

    except Exception as e:
        log.exception("ASK error, fallback: %s", e)
        log_qa({"type": "ask", "qid": qid, "question": question,
                "error": str(e), "answer": "A"})
        return AskResponse(answer="A", sources=sources)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, workers=1)
