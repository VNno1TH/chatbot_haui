"""
retriever.py  (v2 — model nhẹ hơn)
Hybrid Search = Vector (multilingual-e5-small) + BM25 → Reranking (cross-encoder).

THAY ĐỔI so với v1:
  - Embedding model: bge-m3 (570MB) → multilingual-e5-small (118MB)
  - Tắt reranker mặc định (use_reranker=False) — RRF đã đủ tốt, tiết kiệm ~500ms
  - Tăng VECTOR_TOP_K và BM25_TOP_K một chút để bù cho embedding model nhẹ hơn
  - Giữ nguyên: BM25, RRF, parallel search, retrieve cache

Luồng xử lý:
    Query
      ↓
  ┌──────────────────────┐  ┌────────────────────────┐  (SONG SONG)
  │ Vector (e5-small)    │  │ BM25 (rank_bm25)       │
  └──────────────────────┘  └────────────────────────┘
      ↓ Reciprocal Rank Fusion (RRF) — merge + loại trùng
  ┌──────────────────────────────────────┐
  │ [Tuỳ chọn] Cross-encoder Reranker   │
  └──────────────────────────────────────┘
      ↓ Top-k → Context cho LLM
"""

from __future__ import annotations

import re
import math
import time
import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import dataclass, field

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.indexing.embedder import Embedder
from src.pipeline.router   import IntentType

logger = logging.getLogger("haui.retriever")

# ── Cấu hình ─────────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).resolve().parent.parent.parent
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore" / "chroma_db"

# Đọc từ env — mặc định multilingual-e5-small (local CPU)
# Để dùng bge-m3 trên GPU cloud: thêm EMBEDDING_MODEL=BAAI/bge-m3 vào .env
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# Tăng top_k so với v1 vì model nhẹ hơn có recall thấp hơn một chút
VECTOR_TOP_K     = 10   # v1: 7
BM25_TOP_K       = 7    # v1: 5
FINAL_TOP_K      = 4    # giữ nguyên — context cho LLM
RERANKER_INPUT_K = 12   # v1: 10
MIN_RERANK_SCORE = 0.01
RRF_K            = 30

# Reranker — dùng khi cần chất lượng cao hơn, nhưng tốn ~500ms-2s trên CPU
RERANKER_MODEL   = "BAAI/bge-reranker-v2-m3"
RERANK_CACHE_TTL = 600  # 10 phút

INTENT_FILTERS: dict[IntentType, dict | None] = {
    IntentType.RAG_MO_TA_NGANH    : None,
    IntentType.RAG_FAQ             : None,
    IntentType.RAG_TRUONG_HOC_BONG : None,
    IntentType.UNKNOWN             : None,
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    text        : str
    source      : str
    section     : str
    score       : float
    vector_score: float = 0.0
    bm25_score  : float = 0.0
    rrf_score   : float = 0.0
    metadata    : dict  = field(default_factory=dict)


# ── BM25 Index ────────────────────────────────────────────────────────────────

class BM25Index:
    def __init__(self):
        self._bm25      = None
        self._chunks    = []
        self._tokenizer = self._build_tokenizer()

    def _build_tokenizer(self):
        def tokenize(text: str) -> list[str]:
            text = text.lower()
            return re.findall(
                r'[a-záàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+'
                r'|\d+(?:\.\d+)?',
                text
            )
        return tokenize

    def build(self, embedder: Embedder):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Cài đặt: pip install rank-bm25")

        results = embedder.collection.get(include=["documents", "metadatas"])
        if not results["documents"]:
            return

        self._chunks = [
            {"text": doc, "metadata": meta, "id": str(i)}
            for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"]))
        ]
        tokenized    = [self._tokenizer(c["text"]) for c in self._chunks]
        self._bm25   = BM25Okapi(tokenized)
        logger.info(f"BM25 index: {len(self._chunks)} chunks")

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[dict]:
        if self._bm25 is None or not self._chunks:
            return []
        tokens     = self._tokenizer(query)
        scores     = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"text": self._chunks[i]["text"], "metadata": self._chunks[i]["metadata"], "score": float(scores[i])}
            for i in top_indices if scores[i] > 0
        ]

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results  : list[dict],
    k             : int = RRF_K,
) -> list[dict]:
    scores: dict[str, float] = {}
    chunks: dict[str, dict]  = {}

    def _key(c: dict) -> str:
        return c["text"][:100].strip()

    for rank, item in enumerate(vector_results):
        key = _key(item)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in chunks:
            chunks[key] = {**item, "vector_score": item.get("score", 0.0), "bm25_score": 0.0}

    for rank, item in enumerate(bm25_results):
        key = _key(item)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in chunks:
            chunks[key] = {**item, "vector_score": 0.0, "bm25_score": item.get("score", 0.0)}
        else:
            chunks[key]["bm25_score"] = item.get("score", 0.0)

    merged = []
    for key, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        chunk = chunks[key].copy()
        chunk["rrf_score"] = rrf_score
        merged.append(chunk)
    return merged


# ── Cross-encoder Reranker (tuỳ chọn) ────────────────────────────────────────

class CrossEncoderReranker:
    """
    Dùng khi cần chất lượng cao hơn RRF thuần.
    Mặc định TẮT (use_reranker=False trong Retriever) vì tốn ~500ms-2s trên CPU.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        logger.info(f"Loading reranker: {model_name}")
        try:
            from FlagEmbedding import FlagReranker
            self._model    = FlagReranker(model_name, use_fp16=True)
            self._use_flag = True
        except ImportError:
            from sentence_transformers import CrossEncoder
            self._model    = CrossEncoder(model_name)
            self._use_flag = False

        self._cache: dict[str, tuple[float, list[dict]]] = {}
        self._warmup()

    def _warmup(self) -> None:
        dummy = [["câu hỏi tuyển sinh haui", "thông tin tuyển sinh đại học công nghiệp hà nội"]]
        try:
            if self._use_flag:
                self._model.compute_score(dummy, normalize=True)
            else:
                self._model.predict(dummy)
            logger.info("Reranker warm-up xong")
        except Exception as e:
            logger.warning(f"Reranker warm-up thất bại: {e}")

    def _cache_key(self, query: str, chunks: list[dict]) -> str:
        raw = query + "||".join(c["text"][:80] for c in chunks)
        return hashlib.md5(raw.encode("utf-8", errors="ignore")).hexdigest()

    def rerank(self, query: str, chunks: list[dict], top_k: int = FINAL_TOP_K) -> list[dict]:
        if not chunks:
            return []

        key = self._cache_key(query, chunks)
        now = time.monotonic()
        if key in self._cache:
            ts, cached = self._cache[key]
            if now - ts < RERANK_CACHE_TTL:
                return cached[:top_k]

        pairs = [[query, c["text"]] for c in chunks]
        if self._use_flag:
            scores = self._model.compute_score(pairs, normalize=True)
        else:
            scores = self._model.predict([(q, d) for q, d in pairs])

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        self._cache[key] = (now, reranked)

        if len(self._cache) % 100 == 0:
            expired = [k for k, (ts, _) in self._cache.items() if now - ts >= RERANK_CACHE_TTL]
            for k in expired:
                del self._cache[k]

        return reranked[:top_k]


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class Retriever:
    """
    Hybrid retrieval: Vector + BM25 → RRF → (Reranker tuỳ chọn) → top-k.

    Mặc định use_reranker=False vì trên CPU yếu, RRF đã đủ tốt.
    Bật reranker khi cần chất lượng cao hơn và chấp nhận chậm hơn ~500ms.
    """

    _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="retriever")

    def __init__(
        self,
        embedder    : Embedder | None = None,
        use_reranker: bool = False,   # v2: TẮT mặc định
        use_bm25    : bool = True,
    ):
        if embedder:
            self._embedder = embedder
            self._finish_init(use_reranker, use_bm25)
        else:
            self._init_parallel(use_reranker, use_bm25)

    def _init_parallel(self, use_reranker: bool, use_bm25: bool) -> None:
        def _load_embedder():
            return Embedder(
                model_name      = EMBEDDING_MODEL,
                vectorstore_dir = VECTORSTORE_DIR,
            )

        def _load_reranker():
            if not use_reranker:
                return None
            try:
                return CrossEncoderReranker()
            except Exception as e:
                logger.warning(f"Reranker không khả dụng: {e}")
                return None

        fut_embedder = self._executor.submit(_load_embedder)
        fut_reranker = self._executor.submit(_load_reranker)

        self._embedder = fut_embedder.result()
        self._reranker = fut_reranker.result()

        self._bm25_index = None
        if use_bm25:
            logger.info("Building BM25 index...")
            self._bm25_index = BM25Index()
            self._bm25_index.build(self._embedder)

    def _finish_init(self, use_reranker: bool, use_bm25: bool) -> None:
        self._bm25_index = None
        if use_bm25:
            self._bm25_index = BM25Index()
            self._bm25_index.build(self._embedder)

        self._reranker = None
        if use_reranker:
            try:
                self._reranker = CrossEncoderReranker()
            except Exception as e:
                logger.warning(f"Reranker không khả dụng: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query      : str,
        top_k      : int               = FINAL_TOP_K,
        intent_type: IntentType | None = None,
        where      : dict | None       = None,
    ) -> list[RetrievedChunk]:
        # Cache 5 phút
        cache_key = f"{query}||{intent_type}"
        now = time.monotonic()
        if not hasattr(self, "_retrieve_cache"):
            self._retrieve_cache: dict = {}
        if cache_key in self._retrieve_cache:
            ts, cached = self._retrieve_cache[cache_key]
            if now - ts < 300:
                logger.debug("Retrieve cache HIT")
                return cached

        effective_where = where
        if effective_where is None and intent_type is not None:
            effective_where = INTENT_FILTERS.get(intent_type)

        vector_results, bm25_results = self._parallel_search(query, effective_where)

        if bm25_results:
            merged = reciprocal_rank_fusion(vector_results, bm25_results)
        else:
            merged = [{**r, "rrf_score": r["score"]} for r in vector_results]

        if not merged:
            return []

        if self._reranker:
            reranker_input  = merged[:RERANKER_INPUT_K]
            reranked        = self._reranker.rerank(query, reranker_input, top_k=top_k)
            final_score_key = "rerank_score"
        else:
            reranked        = merged[:top_k]
            final_score_key = "rrf_score"

        chunks = []
        for r in reranked:
            meta = r["metadata"]
            chunks.append(RetrievedChunk(
                text         = r["text"],
                source       = meta.get("ten_nganh") or meta.get("source", ""),
                section      = meta.get("section", ""),
                score        = r.get(final_score_key, 0.0),
                vector_score = r.get("vector_score", 0.0),
                bm25_score   = r.get("bm25_score", 0.0),
                rrf_score    = r.get("rrf_score", 0.0),
                metadata     = meta,
            ))

        self._retrieve_cache[cache_key] = (time.monotonic(), chunks)
        return chunks

    def _parallel_search(
        self,
        query          : str,
        effective_where: dict | None,
    ) -> tuple[list[dict], list[dict]]:
        def _vector_search():
            raw = self._embedder.query(
                query_text = query,
                n_results  = VECTOR_TOP_K,
                where      = effective_where,
            )
            return [
                {**r, "vector_score": r["score"], "bm25_score": 0.0}
                for r in raw if r["score"] >= MIN_RERANK_SCORE
            ]

        def _bm25_search():
            if not (self._bm25_index and self._bm25_index.is_ready):
                return []
            raw = self._bm25_index.search(query, top_k=BM25_TOP_K)
            if effective_where:
                raw = [r for r in raw if self._match_filter(r["metadata"], effective_where)]
            return raw

        fut_vector   = self._executor.submit(_vector_search)
        bm25_results = _bm25_search()
        vector_results = fut_vector.result()
        return vector_results, bm25_results

    def retrieve_as_context(
        self,
        query      : str,
        top_k      : int               = FINAL_TOP_K,
        intent_type: IntentType | None = None,
        where      : dict | None       = None,
    ) -> str:
        chunks = self.retrieve(query, top_k, intent_type, where)
        if not chunks:
            return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."

        parts = []
        for chunk in chunks:
            header = (
                f"[{chunk.source} — {chunk.section}]"
                if chunk.section else f"[{chunk.source}]"
            )
            parts.append(f"{header}\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    def _match_filter(self, metadata: dict, where: dict) -> bool:
        for key, value in where.items():
            if isinstance(value, dict):
                if "$in" in value:
                    if metadata.get(key) not in value["$in"]:
                        return False
            else:
                if metadata.get(key) != value:
                    return False
        return True


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time as _time
    print("Khởi tạo Retriever (e5-small + BM25, không reranker)...")
    t0 = _time.perf_counter()
    retriever = Retriever(use_reranker=False, use_bm25=True)
    print(f"✓ Khởi tạo xong ({(_time.perf_counter()-t0)*1000:.0f}ms)\n")

    tests = [
        ("Ngành CNTT học gì?",              IntentType.RAG_MO_TA_NGANH),
        ("học bổng HaUI như thế nào?",      IntentType.RAG_TRUONG_HOC_BONG),
        ("hướng dẫn đăng ký xét tuyển",     IntentType.RAG_FAQ),
        ("Ngành CNTT học gì?",              IntentType.RAG_MO_TA_NGANH),  # test cache
    ]

    for query, intent in tests:
        print(f"{'─'*60}")
        print(f"Query: {query}")
        t = _time.perf_counter()
        chunks = retriever.retrieve(query, intent_type=intent)
        elapsed = (_time.perf_counter() - t) * 1000
        print(f"  ⏱ {elapsed:.0f}ms  ({len(chunks)} chunks)")
        for i, c in enumerate(chunks, 1):
            print(
                f"  {i}. rrf={c.rrf_score:.4f} "
                f"(vec={c.vector_score:.3f} bm25={c.bm25_score:.2f})"
            )
            print(f"     [{c.source} — {c.section}]")
            print(f"     {c.text[:90].strip()}...")
        print()