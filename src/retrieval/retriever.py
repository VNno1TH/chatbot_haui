"""
retriever.py  (v5 — HyDE + Query Rewriting + Self-reflection)

Thay đổi kiến trúc so với v4:

  [V5-1] Query Rewriting (LLM)
         Trước khi retrieve, dùng LLM viết lại query của user thành
         dạng chuẩn hóa, đầy đủ context. 
         "có bao nhiêu cơ sở" → "số lượng cơ sở đào tạo Đại học Công nghiệp Hà Nội HaUI"
         Giải quyết dứt điểm vấn đề câu hỏi tự nhiên không match keyword.

  [V5-2] HyDE (Hypothetical Document Embedding)
         Thay vì embed câu hỏi rồi tìm chunk gần nhất,
         cho LLM viết một câu trả lời giả → embed câu trả lời đó.
         Câu trả lời giả "gần" với chunk thật hơn câu hỏi nhiều.
         Đặc biệt hiệu quả với câu hỏi mô tả/thông tin.

  [V5-3] Self-reflection
         Sau khi retrieve xong, LLM tự đánh giá:
         "Context này có đủ để trả lời câu hỏi không?"
         Nếu không đủ → tự động retry với query khác.
         Tránh LLM "bịa" khi context thiếu.

  Backward compatible với v4:
    - retrieve()            → list[RetrievedChunk]
    - retrieve_v2()         → RetrievalResult
    - retrieve_as_context() → str
"""

from __future__ import annotations

import re
import json
import math
import time
import hashlib
import logging
import os
import requests
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.indexing.embedder import Embedder
from src.pipeline.router   import IntentType

logger = logging.getLogger("haui.retriever")

# ── Cấu hình ──────────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).resolve().parent.parent.parent
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore" / "chroma_db"
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11435")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

USE_REMOTE_RERANKER = os.environ.get("USE_REMOTE_RERANKER", "0") == "1"
RERANKER_URL        = os.environ.get("RERANKER_URL", "http://localhost:8080")

VECTOR_TOP_K     = 10
BM25_TOP_K       = 7
FINAL_TOP_K      = 6
RERANKER_INPUT_K = 12
MIN_RERANK_SCORE = 0.01
RRF_K            = 30
RERANKER_MODEL   = "BAAI/bge-reranker-v2-m3"
RERANK_CACHE_TTL = 600

# V5 config
HYDE_ENABLED           = os.getenv("HYDE_ENABLED", "1") == "1"
QUERY_REWRITE_ENABLED  = os.getenv("QUERY_REWRITE_ENABLED", "1") == "1"
SELF_REFLECT_ENABLED   = os.getenv("SELF_REFLECT_ENABLED", "1") == "1"
LLM_TIMEOUT            = float(os.getenv("RETRIEVER_LLM_TIMEOUT", "6"))

FACTUAL_KEYWORDS = [
    # Cũ — giữ nguyên
    "mã trường", "địa chỉ", "cơ sở", "thành lập", "lịch sử",
    "viết tắt", "tên trường", "haui là", "website", "điện thoại",
    "trực thuộc", "loại hình", "cơ cấu", "tổ chức", "thành viên",
    "có bao nhiêu", "có mấy", "gồm những", "các trường", "các khoa",
    "các viện", "các trung tâm", "sơ đồ", "bộ máy", "đơn vị",
    "bao nhiêu sinh viên", "bao nhiêu giảng viên", "tổng số",
    # Mới — thêm vào ↓
    "trường trực thuộc", "trường thành viên", "khoa trực thuộc",
    "cơ cấu đào tạo", "đơn vị trực thuộc", "trường nào",
    "khoa nào", "viện nào",
    "giới thiệu trường", "giới thiệu haui", "giới thiệu chung",
    "diện tích", "cơ sở vật chất", "ký túc xá", "thư viện",
    "phòng lab", "xưởng thực hành",
    "huân chương", "danh hiệu", "thành tích",
    "giảng viên", "tiến sĩ", "phó giáo sư",
    "tuyển sinh", "lệ phí", "đăng ký xét tuyển",
]

INTENT_FILTERS: dict[IntentType, dict | None] = {
    IntentType.RAG_MO_TA_NGANH    : None,
    IntentType.RAG_FAQ             : None,
    IntentType.RAG_TRUONG_HOC_BONG : None,
    IntentType.UNKNOWN             : None,
}

SCORE_BONUS_NUMERIC     = 0.08
SCORE_BONUS_YEAR        = 0.05
SCORE_BONUS_MONEY       = 0.06
SCORE_BONUS_EXACT_MATCH = 0.10

FALLBACK_RELAXED_THRESHOLD = 0.005
FALLBACK_BROADER_TOP_K     = 6


# ═══════════════════════════════════════════════════════════════════════════════
# [V5-1] QUERY REWRITER — LLM-based
# ═══════════════════════════════════════════════════════════════════════════════

_REWRITE_SYSTEM = """\
Bạn là công cụ chuẩn hóa câu hỏi tuyển sinh HaUI.
Viết lại câu hỏi thành dạng đầy đủ, rõ ràng, có đủ context để tìm kiếm tài liệu.
Giữ nguyên ý nghĩa. Mở rộng viết tắt. Thêm "HaUI" nếu chưa có.
Trả về CHỈ câu hỏi đã viết lại, không giải thích, không dấu ngoặc kép."""

_REWRITE_EXAMPLES = [
    ("trường trực thuộc haui", 
     "các trường thành viên trực thuộc Đại học Công nghiệp Hà Nội HaUI gồm những gì"),
    ("haui có bao nhiêu trường",
     "Đại học Công nghiệp Hà Nội HaUI có bao nhiêu trường thành viên và khoa trực thuộc"),
    ("cơ cấu tổ chức haui",
     "cơ cấu tổ chức đào tạo Đại học Công nghiệp Hà Nội HaUI gồm những đơn vị trường khoa viện nào"),
    ("giới thiệu haui",
     "thông tin chung giới thiệu Đại học Công nghiệp Hà Nội HaUI lịch sử thành tích"),
    ("cơ cấu tổ chức", "cơ cấu tổ chức đào tạo của Đại học Công nghiệp Hà Nội HaUI gồm những đơn vị nào"),
    ("có bao nhiêu cơ sở", "Đại học Công nghiệp Hà Nội HaUI có bao nhiêu cơ sở đào tạo"),
    ("trường thành viên haui", "các trường thành viên trực thuộc Đại học Công nghiệp Hà Nội HaUI"),
    ("học bổng cần gì", "điều kiện xét học bổng tại Đại học Công nghiệp Hà Nội HaUI"),
    ("cntt học gì", "ngành Công nghệ thông tin tại HaUI học những môn gì, chương trình đào tạo"),
]


class QueryRewriter:
    """
    Dùng LLM để viết lại query trước khi đưa vào retrieval.
    
    Tại sao hiệu quả:
    - User hỏi "cơ cấu tổ chức" → BM25 tìm "cơ cấu tổ chức"
    - LLM viết thành "cơ cấu tổ chức đào tạo Đại học Công nghiệp Hà Nội HaUI"
    - BM25 tìm thêm "đại học", "công nghiệp", "hà nội" → match rộng hơn nhiều
    - Vector search: embedding của câu đầy đủ gần với chunk hơn câu ngắn
    
    Cache kết quả để không gọi LLM 2 lần cùng query.
    Timeout ngắn (6s) → nếu fail, dùng query gốc.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self._base_url = base_url.rstrip("/")
        self._model    = model
        self._cache: dict[str, tuple[float, str]] = {}
        self._enabled  = QUERY_REWRITE_ENABLED

    def rewrite(self, query: str) -> str:
        """
        Viết lại query. Trả về query đã viết lại, hoặc query gốc nếu fail.
        """
        if not self._enabled:
            return query

        # Không cần rewrite câu dài (đã đủ context) hoặc câu có nhiều từ khoá cụ thể
        if len(query.split()) >= 12:
            return query

        # Cache check
        cache_key = query.lower().strip()
        now = time.monotonic()
        if cache_key in self._cache:
            ts, rewritten = self._cache[cache_key]
            if now - ts < 600:   # 10 phút TTL
                return rewritten

        # Build few-shot examples
        examples_msgs = []
        for orig, rewritten_ex in _REWRITE_EXAMPLES[:3]:   # chỉ lấy 3 ví dụ để giảm token
            examples_msgs.append({"role": "user",      "content": orig})
            examples_msgs.append({"role": "assistant", "content": rewritten_ex})
        examples_msgs.append({"role": "user", "content": query})

        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": _REWRITE_SYSTEM},
                        *examples_msgs,
                    ],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 60},
                },
                timeout=LLM_TIMEOUT,
            )
            resp.raise_for_status()
            rewritten = resp.json()["message"]["content"].strip()

            # Sanity check: nếu LLM trả về quá dài hoặc quá khác → dùng gốc
            if len(rewritten) > 300 or len(rewritten) < 5:
                return query

            # Lọc bỏ dấu ngoặc kép nếu LLM thêm vào
            rewritten = rewritten.strip('"\'')

            self._cache[cache_key] = (now, rewritten)
            if rewritten.lower() != query.lower():
                logger.debug(f"QueryRewriter: '{query}' → '{rewritten}'")
            return rewritten

        except Exception as e:
            logger.debug(f"QueryRewriter failed ({e}), using original query")
            return query


# ═══════════════════════════════════════════════════════════════════════════════
# [V5-2] HyDE — Hypothetical Document Embedding
# ═══════════════════════════════════════════════════════════════════════════════

_HYDE_SYSTEM = """\
Bạn là chuyên gia tuyển sinh HaUI. Viết một đoạn văn ngắn (3-4 câu) trả lời câu hỏi sau
như thể bạn đang trích dẫn từ tài liệu chính thức của HaUI.
Chỉ viết đoạn văn, không giải thích, không tiêu đề."""


class HyDERetriever:
    """
    Hypothetical Document Embedding.
    
    Thay vì: embed(câu_hỏi) → tìm chunk gần nhất
    Làm thế này: LLM viết câu_trả_lời_giả → embed(câu_trả_lời) → tìm chunk gần nhất
    
    Tại sao tốt hơn:
    - Câu hỏi: "HaUI có bao nhiêu cơ sở?" → embedding xa với chunk mô tả cơ sở
    - Câu trả lời giả: "HaUI có 3 cơ sở đào tạo: cơ sở 1 tại Minh Khai..." 
      → embedding gần với chunk mô tả cơ sở hơn nhiều
    
    Dùng song song với vector search bình thường → merge kết quả.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self._base_url = base_url.rstrip("/")
        self._model    = model
        self._cache: dict[str, tuple[float, str]] = {}
        self._enabled  = HYDE_ENABLED

    def generate_hypothesis(self, query: str) -> Optional[str]:
        """
        Tạo câu trả lời giả cho query.
        Trả None nếu fail → caller sẽ bỏ qua HyDE.
        """
        if not self._enabled:
            return None

        # HyDE hiệu quả nhất với câu hỏi mô tả/thông tin, ít hiệu quả với câu hỏi có số liệu cụ thể
        # Không dùng HyDE cho câu hỏi đã rõ ràng về số (điểm, tiền)
        if re.search(r"\b\d{2}[.,]\d\b", query):
            return None

        cache_key = query.lower().strip()
        now = time.monotonic()
        if cache_key in self._cache:
            ts, hypo = self._cache[cache_key]
            if now - ts < 300:
                return hypo

        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": _HYDE_SYSTEM},
                        {"role": "user",   "content": query},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 120},
                },
                timeout=LLM_TIMEOUT,
            )
            resp.raise_for_status()
            hypothesis = resp.json()["message"]["content"].strip()

            if len(hypothesis) < 20:
                return None

            self._cache[cache_key] = (now, hypothesis)
            logger.debug(f"HyDE generated: '{hypothesis[:80]}...'")
            return hypothesis

        except Exception as e:
            logger.debug(f"HyDE failed ({e})")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# [V5-3] SELF-REFLECTION — Đánh giá context trước khi trả về
# ═══════════════════════════════════════════════════════════════════════════════

_REFLECT_SYSTEM = """\
Đánh giá xem context có đủ để trả lời câu hỏi không.
Trả về JSON: {"sufficient": true/false, "reason": "ngắn gọn", "retry_query": "query khác nếu cần"}
Chỉ trả JSON, không giải thích thêm."""


class SelfReflector:
    """
    Sau khi retrieve xong, LLM tự hỏi: "Context này có đủ không?"
    
    Nếu không đủ → trả về retry_query khác để thử lại.
    Giới hạn 1 lần retry để tránh vòng lặp vô hạn.
    
    Ví dụ:
    - Query: "trường thành viên HaUI"
    - Context: [chunk về học bổng, chunk về điểm chuẩn] → không liên quan
    - Reflect: sufficient=False, retry_query="các đơn vị trực thuộc HaUI trường khoa viện"
    - Retry với query mới → tìm được chunk đúng
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self._base_url = base_url.rstrip("/")
        self._model    = model
        self._enabled  = SELF_REFLECT_ENABLED

    def evaluate(self, query: str, context: str) -> tuple[bool, str]:
        """
        Đánh giá context.
        Trả về (is_sufficient, retry_query).
        retry_query rỗng nếu sufficient=True.
        """
        if not self._enabled or not context or len(context) < 50:
            return True, ""

        try:
            prompt = f"Câu hỏi: {query}\n\nContext:\n{context[:800]}"   # cắt để giảm token
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": _REFLECT_SYSTEM},
                        {"role": "user",   "content": prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 80},
                },
                timeout=LLM_TIMEOUT,
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"].strip()

            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            json_match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
            if not json_match:
                return True, ""

            data = json.loads(json_match.group())
            sufficient  = bool(data.get("sufficient", True))
            retry_query = str(data.get("retry_query", "")).strip()

            if not sufficient:
                logger.debug(f"SelfReflect: NOT sufficient. retry='{retry_query}'")

            return sufficient, retry_query

        except Exception as e:
            logger.debug(f"SelfReflect failed ({e}), assuming sufficient")
            return True, ""


# ═══════════════════════════════════════════════════════════════════════════════
# Giữ nguyên từ v4: QueryExpander, ChunkScorer, FallbackStrategy, data classes,
# BM25Index, RRF, RemoteReranker, CrossEncoderReranker
# ═══════════════════════════════════════════════════════════════════════════════

class QueryExpander:
    """Dictionary-based expansion — vẫn dùng bổ sung cho BM25."""

    _ABBREVIATIONS: dict[str, str] = {
        r"\bcntt\b": "công nghệ thông tin",
        r"\bktđt\b": "kỹ thuật điện tử",
        r"\bqtkd\b": "quản trị kinh doanh",
        r"\bck\b"  : "cơ khí",
        r"\bhaui\b": "đại học công nghiệp hà nội",
        r"\bhui\b" : "đại học công nghiệp hà nội",
    }

    _SYNONYMS: dict[str, list[str]] = {
        "điểm chuẩn"  : ["điểm trúng tuyển", "điểm xét tuyển"],
        "học phí"     : ["chi phí học tập", "tiền học"],
        "tuyển sinh"  : ["xét tuyển", "nhập học", "đầu vào"],
        "ngành"       : ["chuyên ngành", "chương trình đào tạo"],
        "học bổng"    : ["hỗ trợ tài chính", "miễn giảm học phí"],
        "tổ hợp"      : ["khối thi", "môn thi"],
        "chỉ tiêu"    : ["số lượng tuyển"],
    }
    _SYNONYMS_ADDITIONAL = {
    "trực thuộc"   : ["thành viên", "trường con", "đơn vị"],
    "cơ cấu"       : ["tổ chức", "bộ máy", "cấu trúc"],
    "trường"       : ["khoa", "viện", "đơn vị đào tạo"],
    "giới thiệu"   : ["thông tin chung", "tổng quan", "lịch sử"],
    "cơ sở"        : ["campus", "địa điểm", "khuôn viên"],
}

    def expand(self, query: str) -> str:
        expanded = query.lower()
        for pattern, replacement in self._ABBREVIATIONS.items():
            if re.search(pattern, expanded, re.IGNORECASE):
                expanded = re.sub(pattern, f"\\g<0> {replacement}", expanded, flags=re.IGNORECASE)

        extra_terms = []
        for keyword, synonyms in self._SYNONYMS.items():
            if keyword in expanded:
                for syn in synonyms:
                    if syn not in expanded:
                        extra_terms.append(syn)

        if extra_terms:
            expanded = expanded + " " + " ".join(extra_terms)

        seen, result_words = set(), []
        for word in expanded.split():
            if word not in seen:
                seen.add(word)
                result_words.append(word)
        return " ".join(result_words)

    def should_expand(self, query: str) -> bool:
        word_count = len(query.split())
        has_abbrev = any(re.search(p, query, re.IGNORECASE) for p in self._ABBREVIATIONS)
        return word_count <= 10 or has_abbrev


class ChunkScorer:
    _PATTERN_NUMERIC = re.compile(
        r'\b\d+[,.]?\d*\s*(?:điểm|%|triệu|nghìn|tháng|năm|tín chỉ|học kỳ)\b'
        r'|\b\d{4}\b'
        r'|\b\d+\.\d+\b',
        re.IGNORECASE
    )
    _PATTERN_YEAR  = re.compile(r'\b(202[3-9]|203[0-5])\b|\bnăm học\b|\btuyển sinh năm\b', re.IGNORECASE)
    _PATTERN_MONEY = re.compile(r'\b\d+[\.,]?\d*\s*(?:triệu|nghìn|đồng|vnd)\b|\bhọc phí\b.*?\d+', re.IGNORECASE)

    def compute_bonus(self, chunk_text: str, query: str) -> float:
        bonus      = 0.0
        text_lower = chunk_text.lower()

        numeric_matches = self._PATTERN_NUMERIC.findall(chunk_text)
        if len(numeric_matches) >= 2:
            bonus += SCORE_BONUS_NUMERIC
        elif len(numeric_matches) == 1:
            bonus += SCORE_BONUS_NUMERIC * 0.5

        if self._PATTERN_YEAR.search(chunk_text):
            bonus += SCORE_BONUS_YEAR

        if self._PATTERN_MONEY.search(chunk_text):
            bonus += SCORE_BONUS_MONEY

        query_words = set(re.findall(r'\w{3,}', query.lower()))   # ← hạ từ 4 xuống 3
        if query_words:
            matched     = sum(1 for w in query_words if w in text_lower)
            match_ratio = matched / len(query_words)
            if match_ratio >= 0.7:
                bonus += SCORE_BONUS_EXACT_MATCH
            elif match_ratio >= 0.4:
                bonus += SCORE_BONUS_EXACT_MATCH * 0.5

        # Bonus cho block có cấu trúc danh sách (cơ cấu tổ chức, v.v.)
        bullet_count = len(re.findall(r'^\s*[-•*]', chunk_text, re.MULTILINE))
        colon_count  = len(re.findall(r':\s', chunk_text))
        if bullet_count >= 3 or colon_count >= 3:
            bonus += 0.05

        return min(bonus, 0.35)

    def apply_bonus(self, merged_chunks: list[dict], query: str) -> list[dict]:
        for chunk in merged_chunks:
            bonus = self.compute_bonus(chunk.get("text", ""), query)
            chunk["chunk_bonus"]       = bonus
            chunk["rrf_score_boosted"] = chunk.get("rrf_score", 0.0) + bonus

        merged_chunks.sort(key=lambda x: x["rrf_score_boosted"], reverse=True)
        for chunk in merged_chunks:
            chunk["rrf_score"] = chunk["rrf_score_boosted"]
        return merged_chunks


@dataclass
class RetrievalResult:
    chunks        : list
    tier          : int  = 1
    is_miss       : bool = False
    miss_reason   : str  = ""
    expanded_query: str  = ""
    original_query: str  = ""
    rewritten_query: str = ""    # V5 mới
    hyde_used      : bool = False  # V5 mới


class FallbackStrategy:
    MIN_ACCEPTABLE_CHUNKS = 1

    def is_sufficient(self, chunks: list) -> bool:
        return len(chunks) >= self.MIN_ACCEPTABLE_CHUNKS

    def build_miss_result(self, query: str, expanded_query: str, reason: str) -> RetrievalResult:
        return RetrievalResult(
            chunks=[], tier=4, is_miss=True,
            miss_reason=reason, expanded_query=expanded_query, original_query=query,
        )

    def classify_miss_reason(self, query: str, intent_type) -> str:
        q = query.lower()
        if any(w in q for w in ["học phí", "tiền học"]):
            return "no_hoc_phi_data"
        if any(w in q for w in ["điểm chuẩn", "điểm trúng tuyển"]):
            return "no_diem_chuan_data"
        if any(w in q for w in ["học bổng"]):
            return "no_hoc_bong_data"
        if intent_type == IntentType.RAG_MO_TA_NGANH:
            return "nganh_not_found"
        return "general_miss"


@dataclass
class RetrievedChunk:
    text        : str
    source      : str
    section     : str
    score       : float
    vector_score: float = 0.0
    bm25_score  : float = 0.0
    rrf_score   : float = 0.0
    chunk_bonus : float = 0.0
    metadata    : dict  = field(default_factory=dict)


class BM25Index:
    def __init__(self):
        self._bm25      = None
        self._chunks    = []
        self._tokenizer = lambda text: re.findall(r'\w+', text.lower())

    def build(self, embedder: Embedder):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("pip install rank-bm25")

        results = embedder.collection.get(include=["documents", "metadatas"])
        if not results["documents"]:
            return

        self._chunks = [
            {"text": doc, "metadata": meta, "id": str(i)}
            for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"]))
        ]
        tokenized  = [self._tokenizer(c["text"]) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index: {len(self._chunks)} chunks")

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[dict]:
        if self._bm25 is None or not self._chunks:
            return []
        tokens      = self._tokenizer(query)
        scores      = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"text": self._chunks[i]["text"], "metadata": self._chunks[i]["metadata"], "score": float(scores[i])}
            for i in top_indices if scores[i] > 0
        ]

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None


def reciprocal_rank_fusion(vector_results, bm25_results, k=RRF_K):
    scores: dict[str, float] = {}
    chunks: dict[str, dict]  = {}

    def _key(c): return c["text"][:100].strip()

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


class RemoteReranker:
    def __init__(self, base_url: str = RERANKER_URL):
        self._base_url   = base_url.rstrip("/")
        self._rerank_url = f"{self._base_url}/rerank"
        self._cache: dict[str, tuple[float, list[dict]]] = {}
        self._check_connection()

    def _check_connection(self):
        try:
            req = urllib.request.Request(f"{self._base_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5): pass
            logger.info(f"RemoteReranker OK — {self._base_url}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"RemoteReranker unavailable: {e}")

    def _cache_key(self, query, chunks):
        raw = query + "||".join(c["text"][:80] for c in chunks)
        return hashlib.md5(raw.encode("utf-8", errors="ignore")).hexdigest()

    def rerank(self, query: str, chunks: list[dict], top_k: int = FINAL_TOP_K) -> list[dict]:
        if not chunks: return []
        key = self._cache_key(query, chunks)
        now = time.monotonic()
        if key in self._cache:
            ts, cached = self._cache[key]
            if now - ts < RERANK_CACHE_TTL:
                return cached[:top_k]

        payload = json.dumps({"query": query, "texts": [c["text"] for c in chunks], "truncate": True}).encode()
        try:
            req = urllib.request.Request(self._rerank_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            logger.error(f"RemoteReranker error: {e}")
            return chunks[:top_k]

        if isinstance(data, list) and data and isinstance(data[0], dict):
            scored = [(item["index"], item["score"]) for item in data]
        elif isinstance(data, list) and data and isinstance(data[0], (int, float)):
            scored = [(i, s) for i, s in enumerate(data)]
        else:
            return chunks[:top_k]

        for idx, score in scored:
            if idx < len(chunks):
                chunks[idx]["rerank_score"] = float(score)

        reranked = sorted([c for c in chunks if "rerank_score" in c], key=lambda x: x["rerank_score"], reverse=True)
        self._cache[key] = (now, reranked)
        return reranked[:top_k]


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        logger.info(f"Loading local reranker: {model_name}")
        try:
            from FlagEmbedding import FlagReranker
            self._model    = FlagReranker(model_name, use_fp16=True)
            self._use_flag = True
        except ImportError:
            from sentence_transformers import CrossEncoder
            self._model    = CrossEncoder(model_name)
            self._use_flag = False
        self._cache: dict[str, tuple[float, list[dict]]] = {}

    def _cache_key(self, query, chunks):
        raw = query + "||".join(c["text"][:80] for c in chunks)
        return hashlib.md5(raw.encode("utf-8", errors="ignore")).hexdigest()

    def rerank(self, query: str, chunks: list[dict], top_k: int = FINAL_TOP_K) -> list[dict]:
        if not chunks: return []
        key = self._cache_key(query, chunks)
        now = time.monotonic()
        if key in self._cache:
            ts, cached = self._cache[key]
            if now - ts < RERANK_CACHE_TTL:
                return cached[:top_k]

        pairs  = [[query, c["text"]] for c in chunks]
        scores = self._model.compute_score(pairs, normalize=True) if self._use_flag else self._model.predict([(q, d) for q, d in pairs])
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        self._cache[key] = (now, reranked)
        return reranked[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# HYBRID RETRIEVER v5
# ═══════════════════════════════════════════════════════════════════════════════

class Retriever:
    """
    Hybrid Retrieval v5: Query Rewriting + HyDE + Vector + BM25 → RRF → 
                         ChunkScorer → Reranker → Self-reflection → top-k

    Backward compatible:
      retrieve()            → list[RetrievedChunk]
      retrieve_v2()         → RetrievalResult
      retrieve_as_context() → str
    """

    _executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="retriever")

    def __init__(self, embedder=None, use_reranker: bool = True, use_bm25: bool = True):
        # V5 components
        self._query_rewriter = QueryRewriter()
        self._hyde           = HyDERetriever()
        self._reflector      = SelfReflector()

        # V4 components
        self._expander = QueryExpander()
        self._scorer   = ChunkScorer()
        self._fallback = FallbackStrategy()

        if embedder:
            self._embedder = embedder
            self._finish_init(use_reranker, use_bm25)
        else:
            self._init_parallel(use_reranker, use_bm25)

    def _init_parallel(self, use_reranker, use_bm25):
        def _load_embedder():
            return Embedder(model_name=EMBEDDING_MODEL, vectorstore_dir=VECTORSTORE_DIR)
        def _load_reranker():
            return self._build_reranker() if use_reranker else None

        fut_embedder  = self._executor.submit(_load_embedder)
        fut_reranker  = self._executor.submit(_load_reranker)
        self._embedder = fut_embedder.result()
        self._reranker = fut_reranker.result()
        self._bm25_index = None
        if use_bm25:
            self._bm25_index = BM25Index()
            self._bm25_index.build(self._embedder)

    def _finish_init(self, use_reranker, use_bm25):
        self._bm25_index = None
        if use_bm25:
            self._bm25_index = BM25Index()
            self._bm25_index.build(self._embedder)
        self._reranker = self._build_reranker() if use_reranker else None

    def _build_reranker(self):
        if USE_REMOTE_RERANKER:
            try:
                return RemoteReranker(base_url=RERANKER_URL)
            except RuntimeError as e:
                logger.warning(f"Remote reranker fail: {e} — fallback CPU")
        try:
            return CrossEncoderReranker()
        except Exception as e:
            logger.warning(f"Local reranker fail: {e} — RRF only")
            return None

    def _parallel_search(self, query, effective_where, expanded_query=None,
                         vector_top_k=VECTOR_TOP_K, bm25_top_k=BM25_TOP_K,
                         min_score=MIN_RERANK_SCORE):
        bm25_query = expanded_query or query

        def _vector():
            raw = self._embedder.query(query_text=query, n_results=vector_top_k, where=effective_where)
            return [{**r, "vector_score": r["score"], "bm25_score": 0.0} for r in raw if r["score"] >= min_score]

        def _bm25():
            if not (self._bm25_index and self._bm25_index.is_ready):
                return []
            raw = self._bm25_index.search(bm25_query, top_k=bm25_top_k)
            if effective_where:
                raw = [r for r in raw if self._match_filter(r["metadata"], effective_where)]
            return raw

        fut_vec      = self._executor.submit(_vector)
        bm25_results = _bm25()
        return fut_vec.result(), bm25_results

    def _merge_and_score(self, vector_results, bm25_results, query, top_k):
        if bm25_results:
            merged = reciprocal_rank_fusion(vector_results, bm25_results)
        else:
            merged = [{**r, "rrf_score": r["score"]} for r in vector_results]

        if not merged:
            return []

        merged = self._scorer.apply_bonus(merged, query)

        if self._reranker:
            reranked        = self._reranker.rerank(query, merged[:RERANKER_INPUT_K], top_k=top_k)
            final_score_key = "rerank_score"
        else:
            reranked        = merged[:top_k]
            final_score_key = "rrf_score"

        return [
            RetrievedChunk(
                text         = r["text"],
                source       = r["metadata"].get("ten_nganh") or r["metadata"].get("source", ""),
                section      = r["metadata"].get("section", ""),
                score        = r.get(final_score_key, 0.0),
                vector_score = r.get("vector_score", 0.0),
                bm25_score   = r.get("bm25_score", 0.0),
                rrf_score    = r.get("rrf_score", 0.0),
                chunk_bonus  = r.get("chunk_bonus", 0.0),
                metadata     = r["metadata"],
            )
            for r in reranked
        ]

    # ── Public API ──────────────────────────────────────────────────────────────

    def retrieve(self, query, top_k=FINAL_TOP_K, intent_type=None, where=None):
        return self.retrieve_v2(query, top_k, intent_type, where).chunks

    def retrieve_v2(self, query, top_k=FINAL_TOP_K, intent_type=None, where=None) -> RetrievalResult:
        """
        V5 retrieve pipeline:
          1. Query Rewriting  → query chuẩn hóa, đầy đủ context
          2. HyDE             → câu trả lời giả để embed (song song)
          3. Hybrid search    → Vector (query gốc + HyDE) + BM25 (query rewritten)
          4. RRF + ChunkScore + Rerank
          5. Self-reflection  → có đủ context không? → retry 1 lần nếu cần
          6. 3-tier fallback  nếu vẫn miss
        """
        cache_key = f"v5||{query}||{intent_type}"
        now = time.monotonic()
        if not hasattr(self, "_retrieve_cache"):
            self._retrieve_cache: dict = {}
        if cache_key in self._retrieve_cache:
            ts, cached = self._retrieve_cache[cache_key]
            if now - ts < 300:
                return cached

        effective_where = where or (INTENT_FILTERS.get(intent_type) if intent_type else None)

        q_lower = query.lower()
        is_factual = any(kw in q_lower for kw in FACTUAL_KEYWORDS)
        effective_vector_top_k = VECTOR_TOP_K + 8 if is_factual else VECTOR_TOP_K

        # [V5-1] Query Rewriting
        rewritten_query = self._query_rewriter.rewrite(query)

        # [V5-2] HyDE — chạy song song với query rewriting đã xong
        hyde_query = None
        if HYDE_ENABLED:
            fut_hyde = self._executor.submit(self._hyde.generate_hypothesis, query)
        else:
            fut_hyde = None

        # Dictionary expand cho BM25
        expanded_query = self._expander.expand(rewritten_query) if self._expander.should_expand(rewritten_query) else rewritten_query

        # ── Tier 1: Search với rewritten query ────────────────────────────────
        v_res, b_res = self._parallel_search(
            rewritten_query, effective_where,
            expanded_query=expanded_query,
            vector_top_k=effective_vector_top_k,
        )

        # Nếu có HyDE, thêm vector search với hypothesis
        if fut_hyde:
            hypothesis = fut_hyde.result()
            if hypothesis:
                hyde_v_res, _ = self._parallel_search(
                    hypothesis, effective_where,
                    expanded_query=None,
                    vector_top_k=6,
                )
                # Merge HyDE results vào vector results
                v_res = v_res + [r for r in hyde_v_res if r["text"][:80] not in {vr["text"][:80] for vr in v_res}]
                hyde_query = hypothesis

        chunks = self._merge_and_score(v_res, b_res, rewritten_query, top_k)

        if self._fallback.is_sufficient(chunks):
            # [V5-3] Self-reflection
            context_str = "\n".join(c.text[:200] for c in chunks[:3])
            sufficient, retry_q = self._reflector.evaluate(query, context_str)

            if not sufficient and retry_q:
                logger.debug(f"Self-reflect: retrying with '{retry_q}'")
                retry_v, retry_b = self._parallel_search(
                    retry_q, effective_where,
                    expanded_query=self._expander.expand(retry_q),
                    vector_top_k=effective_vector_top_k + 3,
                )
                retry_chunks = self._merge_and_score(retry_v, retry_b, retry_q, top_k)
                if self._fallback.is_sufficient(retry_chunks):
                    chunks = retry_chunks   # retry tốt hơn → dùng

            result = RetrievalResult(
                chunks=chunks, tier=1,
                expanded_query=expanded_query, original_query=query,
                rewritten_query=rewritten_query, hyde_used=hyde_query is not None,
            )
            self._retrieve_cache[cache_key] = (time.monotonic(), result)
            return result

        logger.debug(f"Tier 1 miss for '{query}' — trying Tier 2")

        # ── Tier 2: Relaxed threshold ──────────────────────────────────────────
        v_res2, b_res2 = self._parallel_search(
            rewritten_query, effective_where, expanded_query=expanded_query,
            min_score=FALLBACK_RELAXED_THRESHOLD, vector_top_k=VECTOR_TOP_K + 3,
        )
        chunks2 = self._merge_and_score(v_res2, b_res2, rewritten_query, top_k)

        if self._fallback.is_sufficient(chunks2):
            result = RetrievalResult(
                chunks=chunks2, tier=2,
                expanded_query=expanded_query, original_query=query,
                rewritten_query=rewritten_query,
            )
            self._retrieve_cache[cache_key] = (time.monotonic(), result)
            return result

        logger.debug(f"Tier 2 miss — trying Tier 3")

        # ── Tier 3: Broader expanded ───────────────────────────────────────────
        v_res3, b_res3 = self._parallel_search(
            expanded_query, effective_where, expanded_query=expanded_query,
            vector_top_k=FALLBACK_BROADER_TOP_K + VECTOR_TOP_K,
            bm25_top_k=BM25_TOP_K + 3, min_score=FALLBACK_RELAXED_THRESHOLD,
        )
        chunks3 = self._merge_and_score(v_res3, b_res3, expanded_query, top_k)

        if self._fallback.is_sufficient(chunks3):
            result = RetrievalResult(
                chunks=chunks3, tier=3,
                expanded_query=expanded_query, original_query=query,
                rewritten_query=rewritten_query,
            )
            self._retrieve_cache[cache_key] = (time.monotonic(), result)
            return result

        # ── Tier 4: Miss ───────────────────────────────────────────────────────
        miss_reason = self._fallback.classify_miss_reason(query, intent_type)
        logger.warning(f"Retriever MISS — query='{query}', reason={miss_reason}")
        result = self._fallback.build_miss_result(query, expanded_query, miss_reason)
        self._retrieve_cache[cache_key] = (time.monotonic(), result)
        return result

    def retrieve_as_context(self, query, top_k=FINAL_TOP_K, intent_type=None, where=None) -> str:
        result = self.retrieve_v2(query, top_k, intent_type, where)
        if result.is_miss:
            return self._build_miss_context(result)

        parts = []
        for chunk in result.chunks:
            header   = f"[{chunk.source} — {chunk.section}]" if chunk.section else f"[{chunk.source}]"
            tier_tag = f" [tier={result.tier}]" if result.tier > 1 else ""
            parts.append(f"{header}{tier_tag}\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    def _build_miss_context(self, result: RetrievalResult) -> str:
        msgs = {
            "no_hoc_phi_data"  : "[RETRIEVAL_MISS: học phí]\nKhông tìm thấy thông tin học phí. Hướng dẫn người dùng liên hệ phòng Đào tạo HaUI.",
            "no_diem_chuan_data": "[RETRIEVAL_MISS: điểm chuẩn]\nKhông tìm thấy điểm chuẩn cho ngành/năm được hỏi.",
            "no_hoc_bong_data"  : "[RETRIEVAL_MISS: học bổng]\nKhông có thông tin học bổng cụ thể. Liên hệ phòng Công tác sinh viên.",
            "nganh_not_found"   : "[RETRIEVAL_MISS: ngành]\nKhông tìm thấy thông tin về ngành được hỏi.",
            "general_miss"      : f"[RETRIEVAL_MISS: general]\nKhông tìm thấy thông tin liên quan. Query: '{result.original_query}'",
        }
        return msgs.get(result.miss_reason, msgs["general_miss"])

    def _match_filter(self, metadata, where):
        for key, value in where.items():
            if isinstance(value, dict):
                if "$in" in value and metadata.get(key) not in value["$in"]:
                    return False
            else:
                if metadata.get(key) != value:
                    return False
        return True

    @property
    def reranker_type(self) -> str:
        if self._reranker is None: return "none (RRF only)"
        if isinstance(self._reranker, RemoteReranker): return f"remote ({RERANKER_URL})"
        return "local CPU"