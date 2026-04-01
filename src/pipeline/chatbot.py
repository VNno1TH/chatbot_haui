"""
chatbot.py  (v6 — Kiến trúc thông minh thật sự)

Thay đổi so với v5:

  [A] Tích hợp Router v3 (LLM-based intent classification)
      → Không còn miss câu tự nhiên vì thiếu keyword

  [B] Tích hợp Retriever v5 (Query Rewriting + HyDE + Self-reflection)
      → Query được chuẩn hóa trước khi search
      → HyDE tạo "câu trả lời giả" để embed, match chunk tốt hơn
      → Self-reflection tự kiểm tra context đủ chưa

  [C] System prompt thông minh — phân nhóm thông tin A/B
      Nhóm A (số liệu): cực nghiêm, không được bịa
      Nhóm B (mô tả): linh hoạt hơn, có thể dùng kiến thức nền có chú thích

  [D] Debug mode — log rõ ràng từng bước để dễ trace lỗi

  Backward compatible: API chat() và chat_stream() không thay đổi
"""

from __future__ import annotations

import os
import re
import json
import time
import random
import hashlib
import logging
import requests
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterator, Optional
from dotenv import load_dotenv

load_dotenv()

# ── Import pipeline components ────────────────────────────────────────────────
# Router v3 — LLM-based
from src.pipeline.router     import Router, Intent, IntentType, _extract_entities_rule

# Retriever v5 — HyDE + Query Rewriting + Self-reflection
from src.retrieval.retriever import Retriever, RetrievalResult

from src.pipeline.profiler   import LatencyProfiler

# JSON data functions (giữ nguyên)
from src.query_json import (
    get_chi_tieu_nganh, get_nganh_theo_to_hop, get_nganh_theo_khoa,
    get_chi_tieu_tong_2026, get_mon_thi_to_hop,
    get_diem_chuan, get_diem_chuan_moi_nhat, get_lich_su_diem_chuan,
    get_diem_chuan_theo_khoa,
    get_hoc_phi,
    quy_doi_HSA, quy_doi_TSA, quy_doi_KQHB,
    get_diem_uu_tien_khu_vuc, get_diem_uu_tien_doi_tuong,
    tinh_diem_uu_tien, kiem_tra_dau_truot,
    fmt_diem_chuan, fmt_diem_chuan_theo_khoa, fmt_nganh_theo_khoa,
    fmt_hoc_phi, fmt_chi_tieu_nganh,
    fmt_nganh_theo_to_hop, fmt_chi_tieu_2026, fmt_mon_thi_to_hop,
    fmt_tinh_diem_uu_tien, fmt_quy_doi, fmt_kiem_tra_dau_truot,
)

# ── [FIX v7] Import entity extractor v2 + nganh alias mở rộng ────────────────
# entity_extractor.py: nhận dạng tên trường/khoa từ câu hỏi tự nhiên
# nganh.py đã được thay bằng nganh_v2 có 80+ alias cho 5 trường + 4 khoa
try:
    from src.query_json.nganh import (
        get_nganh_theo_khoa     as _get_nganh_theo_khoa_v2,
        get_co_cau_truong_khoa,
        _resolve_khoa,
    )
    _NGANH_V2_AVAILABLE = True
except ImportError:
    _NGANH_V2_AVAILABLE = False
    logger_tmp = logging.getLogger("haui.chatbot")
    logger_tmp.warning("nganh_v2 functions not found, falling back to v1")

logger = logging.getLogger("haui.chatbot")

# ── Cấu hình ──────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11435")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
MAX_HISTORY     = 6
MAX_NEW_TOKENS  = 2000
TEMPERATURE     = 0.2
MAX_CONTEXT_CHARS = 2400

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
USE_GROQ     = bool(GROQ_API_KEY)

RESPONSE_CACHE_TTL      = 600
RESPONSE_CACHE_MAX_SIZE = 256
STREAM_BUFFER_CHARS     = 8
STREAM_SENTENCE_FLUSH   = True

# Debug mode — log chi tiết từng bước
DEBUG_MODE = os.getenv("HAUI_DEBUG", "0") == "1"


# ══════════════════════════════════════════════════════════════════════════════
# [C] SYSTEM PROMPT — Phân nhóm A/B, thông minh hơn
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
Bạn là trợ lý tư vấn tuyển sinh Đại học Công nghiệp Hà Nội (HaUI).
Nhiệm vụ: Trả lời dựa HOÀN TOÀN vào [THÔNG TIN THAM KHẢO] bên dưới.
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NGUYÊN TẮC CỐT LÕI — KHÔNG ĐƯỢC VI PHẠM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
1. CHỈ dùng thông tin có trong [THÔNG TIN THAM KHẢO].
   NGHIÊM CẤM dùng kiến thức huấn luyện của bạn về HaUI — dù bạn "chắc chắn".
 
2. Nếu [THÔNG TIN THAM KHẢO] không có câu trả lời → trả lời đúng 1 mẫu:
   "Tôi chưa có thông tin về [X] trong dữ liệu hiện tại.
   Vui lòng xem tại tuyensinh.haui.edu.vn hoặc gọi 0243.7655121."
 
3. KHÔNG suy luận, KHÔNG suy đoán, KHÔNG ngoại suy từ dữ liệu khác.
   Ví dụ: có dữ liệu về Trường CNTT nhưng không có về Khoa Hóa → KHÔNG suy "Khoa Hóa chắc cũng tương tự".
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHÂN LOẠI THÔNG TIN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
NHÓM A — Số liệu chính xác (điểm chuẩn, học phí, chỉ tiêu, mã ngành):
• Trích dẫn đầy đủ, chính xác từng con số. Không làm tròn, không bỏ sót.
• Nếu thiếu → dùng mẫu câu ở nguyên tắc 2.
 
NHÓM B — Thông tin mô tả (cơ cấu tổ chức, ngành học, quy trình):
• Dùng đúng nội dung từ [THÔNG TIN THAM KHẢO], không thêm bớt.
• Nếu thiếu → dùng mẫu câu ở nguyên tắc 2. KHÔNG được thêm "có thể", "thường là", "theo hiểu biết của tôi".
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUY TẮC XÉT TUYỂN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• KHÔNG kết luận "đậu" hay "trượt" — điểm chuẩn 2026 chưa công bố.
• Trình bày: điểm xét → điểm chuẩn tham chiếu → chênh lệch → nhận xét.
• Kết thúc: "Theo dõi thông báo tại tuyensinh.haui.edu.vn."
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NGÔN NGỮ & ĐỊNH DẠNG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• LUÔN LUÔN trả lời bằng TIẾNG VIỆT. Tuyệt đối không dùng tiếng Trung, tiếng Anh hay ngôn ngữ khác.
• Tối đa 8 câu. Không dùng emoji.
• Ký hiệu đúng: KV1, KV2-NT, PT3, HSA, TSA, HaUI.\
"""


# ══════════════════════════════════════════════════════════════════════════════
# Giữ nguyên từ v5: ResponseCache, ConfidenceHedger, StreamBuffer
# ══════════════════════════════════════════════════════════════════════════════

class ResponseCache:
    _CACHEABLE_INTENTS = {
        IntentType.RAG_MO_TA_NGANH,
        IntentType.RAG_FAQ,
        IntentType.RAG_TRUONG_HOC_BONG,
        IntentType.JSON_DIEM_CHUAN,
        IntentType.JSON_HOC_PHI,
        IntentType.JSON_CHI_TIEU_TO_HOP,
    }
    _NO_CACHE_INTENTS = {IntentType.JSON_QUY_DOI_DIEM, IntentType.JSON_DAU_TRUOT}

    def __init__(self, ttl=RESPONSE_CACHE_TTL, max_size=RESPONSE_CACHE_MAX_SIZE):
        self._ttl      = ttl
        self._max_size = max_size
        self._store: OrderedDict[str, tuple[float, str]] = OrderedDict()

    def _make_key(self, query, intent_type, entities):
        q_norm     = re.sub(r'\s+', ' ', query.lower().strip())
        q_norm     = re.sub(r'[?!.,;]', '', q_norm)
        entity_str = json.dumps({k: v for k, v in sorted(entities.items()) if v is not None}, ensure_ascii=False)
        raw        = f"{q_norm}|{intent_type.value}|{entity_str}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, query, intent_type, entities) -> Optional[str]:
        if intent_type in self._NO_CACHE_INTENTS: return None
        key = self._make_key(query, intent_type, entities)
        if key not in self._store: return None
        ts, answer = self._store[key]
        if time.monotonic() - ts > self._ttl:
            del self._store[key]; return None
        self._store.move_to_end(key)
        return answer

    def set(self, query, intent_type, entities, answer):
        if intent_type in self._NO_CACHE_INTENTS: return
        if not answer or len(answer) < 10: return
        if any(e in answer for e in ["[Lỗi", "Xin lỗi, có lỗi"]): return
        key = self._make_key(query, intent_type, entities)
        if key in self._store: del self._store[key]
        if len(self._store) >= self._max_size:
            del self._store[next(iter(self._store))]
        self._store[key] = (time.monotonic(), answer)

    def invalidate(self):
        self._store.clear()

    def stats(self):
        now   = time.monotonic()
        valid = sum(1 for ts, _ in self._store.values() if now - ts <= self._ttl)
        return {"total": len(self._store), "valid": valid, "expired": len(self._store) - valid}


class ConfidenceHedger:
    _EXACT_DATA_INTENTS = {
        IntentType.JSON_DIEM_CHUAN, IntentType.JSON_HOC_PHI,
        IntentType.JSON_CHI_TIEU_TO_HOP, IntentType.JSON_QUY_DOI_DIEM, IntentType.JSON_DAU_TRUOT,
    }
    _OFFICIAL_SOURCES = {
        IntentType.RAG_MO_TA_NGANH    : "tuyensinh.haui.edu.vn",
        IntentType.RAG_FAQ             : "tuyensinh.haui.edu.vn",
        IntentType.RAG_TRUONG_HOC_BONG : "haui.edu.vn",
        IntentType.UNKNOWN             : "tuyensinh.haui.edu.vn",
    }

    def should_hedge(self, intent_type, retrieval_tier, is_miss, intent_confidence) -> bool:
        if intent_type in self._EXACT_DATA_INTENTS: return False
        if is_miss: return True
        if retrieval_tier >= 2: return True
        if intent_confidence < 0.55: return True
        return False

    def build_prefix(self, retrieval_tier, is_miss, miss_reason, intent_confidence) -> str:
        if is_miss:
            msgs = {
                "no_hoc_phi_data"   : "Tôi chưa tìm thấy thông tin học phí chính xác cho câu hỏi này.",
                "no_diem_chuan_data": "Tôi chưa có dữ liệu điểm chuẩn cho năm/ngành được hỏi.",
                "no_hoc_bong_data"  : "Tôi chưa có thông tin học bổng chi tiết.",
                "nganh_not_found"   : "Tôi chưa tìm thấy thông tin về ngành này trong dữ liệu hiện có.",
                "general_miss"      : "Tôi không tìm thấy thông tin chính xác cho câu hỏi này.",
            }
            return msgs.get(miss_reason, msgs["general_miss"])
        if retrieval_tier == 2: return "Thông tin dưới đây có thể chưa đầy đủ — tôi đã mở rộng tìm kiếm:"
        if retrieval_tier == 3: return "Tôi không tìm thấy kết quả chính xác, dưới đây là thông tin liên quan gần nhất:"
        if intent_confidence < 0.55: return "Tôi không chắc chắn về ý của câu hỏi, tôi sẽ cố gắng trả lời:"
        return ""

    def build_suffix(self, intent_type, retrieval_tier, is_miss) -> str:
        source = self._OFFICIAL_SOURCES.get(intent_type)
        if source and (is_miss or retrieval_tier >= 2):
            return f"\nĐể có thông tin chính xác nhất, vui lòng xem tại: {source}"
        return ""

    def apply(self, answer, intent_type, retrieval_tier, is_miss, miss_reason, intent_confidence) -> str:
        if not self.should_hedge(intent_type, retrieval_tier, is_miss, intent_confidence):
            return answer
        prefix = self.build_prefix(retrieval_tier, is_miss, miss_reason, intent_confidence)
        suffix = self.build_suffix(intent_type, retrieval_tier, is_miss)
        parts  = []
        if prefix: parts.append(prefix)
        parts.append(answer)
        if suffix: parts.append(suffix)
        return "\n\n".join(parts)


class StreamBuffer:
    FLUSH_CHARS     = set('!?…\n')
    MAX_BUFFER_CHARS = 80
    _SENTENCE_DOT   = re.compile(r'\.(?=\s|$)')

    def __init__(self, min_chars=STREAM_BUFFER_CHARS):
        self._buf       = ""
        self._min_chars = min_chars

    def feed(self, token) -> Iterator[str]:
        self._buf += token
        if STREAM_SENTENCE_FLUSH:
            flush_pos = -1
            for i, ch in enumerate(self._buf):
                if ch in self.FLUSH_CHARS:
                    flush_pos = i
                elif ch == '.':
                    prev_digit = i > 0 and self._buf[i-1].isdigit()
                    next_digit = i + 1 < len(self._buf) and self._buf[i+1].isdigit()
                    if not prev_digit and not next_digit:
                        flush_pos = i
            if flush_pos >= 0:
                chunk      = self._buf[:flush_pos + 1]
                self._buf  = self._buf[flush_pos + 1:]
                if chunk.strip(): yield chunk
                return

        if len(self._buf) >= self._min_chars:
            yield self._buf; self._buf = ""; return

        if len(self._buf) >= self.MAX_BUFFER_CHARS:
            yield self._buf; self._buf = ""

    def flush(self) -> Iterator[str]:
        if self._buf.strip(): yield self._buf
        self._buf = ""

    def reset(self): self._buf = ""


# ══════════════════════════════════════════════════════════════════════════════
# EntityTracker, QueryRewriter (conversation-level, giữ từ v5)
# ══════════════════════════════════════════════════════════════════════════════

class EntityTracker:
    _PRONOUN_NGANH = re.compile(r"\bngành (đó|này|kia|trên|vừa (nói|hỏi)|đã (nói|hỏi))\b")
    _PRONOUN_DIEM  = re.compile(r"\bđiểm (đó|này|kia|trên|vừa nói)\b")

    def __init__(self):
        self._slots: dict[str, object] = {}

    def update(self, entities: dict):
        for key, val in entities.items():
            if val is not None:
                self._slots[key] = val

    def resolve(self, query: str, entities: dict) -> dict:
        resolved = dict(entities)
        if not resolved.get("nganh") and self._PRONOUN_NGANH.search(query.lower()):
            if "nganh" in self._slots: resolved["nganh"] = self._slots["nganh"]
        if not resolved.get("diem") and self._PRONOUN_DIEM.search(query.lower()):
            if "diem" in self._slots: resolved["diem"] = self._slots["diem"]
        if not resolved.get("khu_vuc") and "khu_vuc" in self._slots:
            if any(kw in query.lower() for kw in ["điểm", "đậu", "trượt", "xét", "ưu tiên"]):
                resolved["khu_vuc"] = self._slots["khu_vuc"]
        if not resolved.get("nganh") and "nganh" in self._slots:
            q_lower   = query.lower()
            word_count = len(q_lower.split())
            continuation = ["thế còn", "vậy còn", "còn ngành", "điểm đó", "học phí đó",
                            "thì sao", "thế nào", "còn học phí", "còn điểm", "ngành đó", "ngành này"]
            tuyensinh_kws = ["điểm chuẩn", "học phí", "tổ hợp", "chỉ tiêu", "xét tuyển"]
            has_cont = any(p in q_lower for p in continuation)
            has_ts   = any(k in q_lower for k in tuyensinh_kws)
            if has_cont or (has_ts and word_count <= 6):
                resolved["nganh"] = self._slots["nganh"]
        return resolved

    def get(self, key, default=None):
        return self._slots.get(key, default)

    def has(self, key):
        return key in self._slots

    @property
    def summary(self):
        return str(self._slots)


class ConversationQueryRewriter:
    """Conversation-level query rewriter (khác với LLM-based QueryRewriter trong retriever)."""
    _SHORT_PATTERNS = [
        re.compile(r"^(thế\s+)?(còn|vậy\s+còn)\s+(học\s*phí|tiền\s*học)"),
        re.compile(r"^(thế\s+)?(còn|vậy\s+còn)\s+(điểm|điểm\s+chuẩn)"),
        re.compile(r"^(thế\s+)?(còn|vậy\s+còn)\s+(ngành|tổ\s+hợp|chỉ\s+tiêu)"),
    ]

    def rewrite(self, query: str, tracker: EntityTracker, recent_history: list) -> str:
        q       = query.strip()
        q_lower = q.lower()
        if len(q.split()) > 7: return q
        nganh = tracker.get("nganh")
        if re.search(r"học\s*phí|tiền\s*học|chi\s*phí", q_lower):
            if nganh and "ngành" not in q_lower:
                return f"Học phí ngành {nganh} là bao nhiêu?"
        if re.search(r"điểm\s*chuẩn|điểm\s*đầu\s*vào", q_lower):
            if nganh and "ngành" not in q_lower:
                return f"Điểm chuẩn ngành {nganh} là bao nhiêu?"
        if re.search(r"tổ\s*hợp|môn\s*thi", q_lower):
            if nganh and "ngành" not in q_lower:
                return f"Ngành {nganh} xét tuyển tổ hợp môn nào?"
        return q


class ContextCompressor:
    _NUMBER_LINE = re.compile(r'\d+[\.,]\d+|\d{4,}|\d+\s*(điểm|đồng|%|tín)')

    def compress(self, context: str, max_chars: int = MAX_CONTEXT_CHARS, query: str = "") -> str:
        if len(context) <= max_chars: return context
        blocks = re.split(r'\n\n+|\n---+\n', context)

        query_keywords: set[str] = set()
        if query:
            query_keywords = {w for w in re.findall(r'\w{3,}', query.lower())}

        scored = []
        for block in blocks:
            lines         = block.split('\n')
            number_lines  = sum(1 for ln in lines if self._NUMBER_LINE.search(ln))
            numeric_score = number_lines * 10

            kw_bonus = 0
            if query_keywords:
                block_lower = block.lower()
                matched     = sum(1 for kw in query_keywords if kw in block_lower)
                kw_bonus    = matched * 8

            # Bonus cho danh sách/bảng
            bullet_count = len(re.findall(r'^\s*[-•*]', block, re.MULTILINE))
            colon_count  = len(re.findall(r':\s', block))
            list_bonus   = 15 if (bullet_count >= 3 or colon_count >= 3) else 0

            length_penalty = len(block) / 100
            score          = numeric_score + kw_bonus + list_bonus - length_penalty
            scored.append((score, block))

        scored.sort(key=lambda x: x[0], reverse=True)
        result_parts, current_len = [], 0
        for _, block in scored:
            if current_len + len(block) + 2 <= max_chars:
                result_parts.append(block)
                current_len += len(block) + 2
            else:
                truncated = block[:max(200, max_chars - current_len - 10)]
                if truncated.strip(): result_parts.append(truncated + "\n  [...]")
                break

        order_map = {b: i for i, b in enumerate(blocks)}
        result_parts.sort(key=lambda b: order_map.get(b.replace("\n  [...]", ""), 999))
        return "\n\n".join(result_parts)


class HistoryManager:
    def __init__(self, max_pairs=MAX_HISTORY):
        self._max_pairs = max_pairs
        self._history  : list = []
        self._summary  : str  = ""

    def add(self, user_msg, assistant_msg):
        self._history.append({"role": "user",      "content": user_msg})
        self._history.append({"role": "assistant", "content": assistant_msg})
        if len(self._history) > self._max_pairs * 2:
            self._compress()

    def _compress(self):
        half      = len(self._history) // 2
        old_pairs = self._history[:half]
        self._history = self._history[half:]
        items = []
        for i in range(0, len(old_pairs) - 1, 2):
            u = old_pairs[i]["content"][:80]
            a = old_pairs[i+1]["content"][:120] if i+1 < len(old_pairs) else ""
            items.append(f"- Hỏi: {u}\n  Đáp: {a}")
        new_summary   = "\n".join(items)
        self._summary = (self._summary + "\n" + new_summary).strip()
        if len(self._summary) > 800:
            lines         = self._summary.split("\n")
            self._summary = "\n".join(lines[-20:])

    def get_messages(self) -> list:
        return self._history[-self._max_pairs * 2:]

    def get_recent(self, n=4) -> list:
        return self._history[-n:]

    def get_summary_prefix(self) -> str:
        if not self._summary: return ""
        return f"\n\n[Tóm tắt cuộc trò chuyện trước:]\n{self._summary}"

    def clear(self):
        self._history.clear()
        self._summary = ""


@dataclass
class Message:
    role   : str
    content: str


@dataclass
class ChatResponse:
    answer          : str
    intent          : IntentType
    method          : str
    context         : str
    confidence      : float
    rewritten_query : str  = ""
    cache_hit       : bool = False
    retrieval_tier  : int  = 1
    hyde_used       : bool = False   # V6 mới
    router_method   : str  = ""      # V6 mới — "rule"/"llm"/"embed"/"fallback"


# ══════════════════════════════════════════════════════════════════════════════
# LLM Wrappers (giữ nguyên từ v5)
# ══════════════════════════════════════════════════════════════════════════════

class OllamaLLM:
    def __init__(self, model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL):
        self._model    = model
        self._base_url = base_url.rstrip("/")
        self._check_connection()

    def _check_connection(self):
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Không kết nối được Ollama. Kiểm tra Ollama đang chạy.")
        except Exception as e:
            raise RuntimeError(f"Lỗi kết nối Ollama: {e}")
        models     = [m["name"] for m in resp.json().get("models", [])]
        model_base = self._model.split(":")[0]
        if not any(model_base in m for m in models):
            raise RuntimeError(f"Model '{self._model}' chưa được tải. Chạy: ollama pull {self._model}")

    def _build_messages(self, system, history, user_msg):
        msgs = [{"role": "system", "content": system}]
        for m in history:
            if isinstance(m, dict):
                msgs.append(m)
            else:
                msgs.append({"role": m.role, "content": m.content})
        msgs.append({"role": "user", "content": user_msg})
        return msgs

    def generate(self, system, history, user_msg) -> str:
        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json={"model": self._model, "messages": self._build_messages(system, history, user_msg),
                      "stream": False, "options": {"temperature": TEMPERATURE, "num_predict": MAX_NEW_TOKENS}},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except requests.exceptions.Timeout:
            return "Xin lỗi, model phản hồi quá chậm. Vui lòng thử lại."
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return "Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi."

    def generate_stream(self, system, history, user_msg) -> Iterator[str]:
        try:
            with requests.post(
                f"{self._base_url}/api/chat",
                json={"model": self._model, "messages": self._build_messages(system, history, user_msg),
                      "stream": True, "options": {"temperature": TEMPERATURE, "num_predict": MAX_NEW_TOKENS}},
                stream=True, timeout=120,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line: continue
                    try:
                        data  = json.loads(line.decode("utf-8"))
                        token = data.get("message", {}).get("content", "")
                        if token: yield token
                        if data.get("done"): break
                    except json.JSONDecodeError:
                        continue
        except requests.exceptions.Timeout:
            yield "\n[Lỗi: model phản hồi quá chậm]"
        except Exception as e:
            logger.error(f"Ollama stream error: {e}")
            yield "\n[Lỗi kết nối Ollama]"


class GroqLLM:
    def __init__(self, model=GROQ_MODEL, api_key=GROQ_API_KEY):
        try:
            from groq import Groq
        except ImportError:
            raise RuntimeError("Chạy: pip install groq")
        self._client = Groq(api_key=api_key)
        self._model  = model

    def _build_messages(self, system, history, user_msg):
        msgs = [{"role": "system", "content": system}]
        for m in history:
            if isinstance(m, dict): msgs.append(m)
            else: msgs.append({"role": m.role, "content": m.content})
        msgs.append({"role": "user", "content": user_msg})
        return msgs

    def generate(self, system, history, user_msg) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._model, messages=self._build_messages(system, history, user_msg),
                max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, stream=False,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq error: {e}")
            return "Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi."

    def generate_stream(self, system, history, user_msg) -> Iterator[str]:
        try:
            stream = self._client.chat.completions.create(
                model=self._model, messages=self._build_messages(system, history, user_msg),
                max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, stream=True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token: yield token
        except Exception as e:
            logger.error(f"Groq stream error: {e}")
            yield "\n[Lỗi kết nối Groq]"


# ══════════════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════════════

def _light_normalize(query: str) -> str:
    q      = query.strip()
    abbrevs = {
        r"\bcntt\b": "công nghệ thông tin",
        r"\bktpm\b": "kỹ thuật phần mềm",
        r"\bqtkd\b": "quản trị kinh doanh",
        r"\bhttt\b": "hệ thống thông tin",
        r"\btmdt\b": "thương mại điện tử",
        r"\btmđt\b": "thương mại điện tử",
    }
    q_lower = q.lower()
    for pattern, replacement in abbrevs.items():
        q_lower = re.sub(pattern, replacement, q_lower)
    QUESTION_WORDS = ("bao nhiêu", "mấy", "như thế nào", "là gì", "ở đâu", "khi nào", "có không", "được không")
    if any(w in q_lower for w in QUESTION_WORDS) and not q.endswith("?"):
        q = q + "?"
    return q


# ── [FIX v7] Entity extraction v2 — nhận dạng tên trường/khoa ────────────────

def _extract_truong_khoa_from_query(query: str) -> Optional[str]:
    """
    Trích xuất tên trường/khoa từ câu hỏi tự nhiên.
    Ví dụ:
      "Trường Cơ khí - Ô tô có bao nhiêu ngành?" -> "Cơ khí - Ô tô"
      "các ngành trong trường kinh tế"            -> "kinh tế"
      "khoa CNTT có những ngành gì"               -> "CNTT"
    """
    _PATTERNS = [
        re.compile(
            r"(?:trường|khoa)\s+(.{2,60}?)\s+(?:có|gồm)\s+(?:bao nhiêu|mấy)\s+ngành",
            re.IGNORECASE | re.UNICODE,
        ),
        re.compile(
            r"(?:trong|thuộc|của)\s+(?:trường|khoa)\s+(.{2,60}?)(?:\s+(?:có|gồm|là|điểm|học phí)|\?|$)",
            re.IGNORECASE | re.UNICODE,
        ),
        re.compile(
            r"(?:trường|khoa)\s+(.{2,60}?)\s+(?:có|gồm|những|các)\s+(?:những\s+)?ngành",
            re.IGNORECASE | re.UNICODE,
        ),
        re.compile(
            r"ngành\s+(?:nào|của|thuộc)\s+(?:trường|khoa)\s+(.{2,60}?)(?:\s*\?|$)",
            re.IGNORECASE | re.UNICODE,
        ),
        re.compile(
            r"(?:trường|khoa)\s+(.{2,60}?)\s+(?:điểm chuẩn|xét tuyển|học phí)",
            re.IGNORECASE | re.UNICODE,
        ),
        re.compile(
            r"(?:vào|đậu|trượt)\s+(?:trường|khoa)\s+(.{2,60}?)(?:\s*\?|$)",
            re.IGNORECASE | re.UNICODE,
        ),
    ]
    _STOP_TRAILING = ["có", "gồm", "là", "thì", "thuộc", "bao nhiêu", "mấy", "nào", "và", "với", "của"]

    for pat in _PATTERNS:
        m = pat.search(query)
        if m:
            raw = m.group(1).strip().rstrip("?.,!")
            for stop in _STOP_TRAILING:
                if raw.lower().endswith(" " + stop):
                    raw = raw[:-(len(stop) + 1)].strip()
            if len(raw) >= 2:
                return raw
    return None


def _extract_entities_v2(query: str) -> dict:
    """
    Entity extraction v2: giữ toàn bộ logic v1, thêm nhận dạng trường/khoa.
    Backward compatible với v1.
    """
    entities = _extract_entities_rule(query)

    ten_truong = _extract_truong_khoa_from_query(query)
    if ten_truong:
        entities["ten_truong"] = ten_truong
        entities["is_truong_query"] = True

    _CO_CAU_PATTERNS = [
        r"bao nhiêu\s+(?:trường|khoa)",
        r"(?:trường|khoa)\s+nào",
        r"(?:có|gồm)\s+(?:những|các)\s+(?:trường|khoa)",
        r"cơ cấu\s+(?:tổ chức|đào tạo)",
        r"các\s+đơn vị\s+trực thuộc",
        r"thành viên\s+(?:của\s+)?haui",
        r"trường\s+trực thuộc",
        r"đơn vị\s+(?:trực thuộc|thành viên)",
        r"sơ đồ\s+tổ chức",
    ]
    q_lower = query.lower()
    if any(re.search(p, q_lower) for p in _CO_CAU_PATTERNS):
        entities["is_co_cau_query"] = True

    return entities


# ══════════════════════════════════════════════════════════════════════════════
# Context Builder (giữ logic từ v5, tích hợp retriever v5)
# ══════════════════════════════════════════════════════════════════════════════

class ContextBuilder:
    def __init__(self, retriever: Retriever):
        self._retriever  = retriever
        self._pending    : dict = {}
        self._compressor = ContextCompressor()
        self._last_retrieval: Optional[RetrievalResult] = None

    def build(self, query: str, intent: Intent, tracker: EntityTracker) -> str:
        if intent.intent_type == IntentType.GREETING:  return "__GREETING__"
        if intent.intent_type == IntentType.OFF_TOPIC: return "__OFF_TOPIC__"

        # [FIX v7] Xử lý câu hỏi về cơ cấu tổ chức tổng quan
        entities = intent.entities
        if entities.get("is_co_cau_query") or any(
            kw in query.lower()
            for kw in [
                "bao nhiêu trường", "bao nhiêu khoa",
                "trường nào", "khoa nào", "cơ cấu tổ chức",
                "trường trực thuộc", "đơn vị trực thuộc",
                "thành viên haui", "trường thành viên",
                "có mấy trường", "có mấy khoa",
            ]
        ):
            co_cau_ctx = self._ctx_co_cau(query)
            rag_ctx    = self._ctx_rag(query, intent, tracker)
            if co_cau_ctx:
                combined = co_cau_ctx + "\n\n---\n\n" + rag_ctx
                return self._compressor.compress(combined, query=query)

        khoa_detected = self._detect_khoa(query)
        if khoa_detected:
            diem = intent.entities.get("diem") or tracker.get("diem")
            kv   = intent.entities.get("khu_vuc") or tracker.get("khu_vuc")
            dt   = intent.entities.get("doi_tuong") or tracker.get("doi_tuong")
            raw_ctx = (self._ctx_dau_truot_theo_khoa(query, diem, khoa_detected, kv, dt)
                       if diem else self._ctx_nganh_theo_khoa(query, khoa_detected))
            if raw_ctx and "__CLARIFY__" not in raw_ctx:
                raw_ctx = self._compressor.compress(raw_ctx, query=query)
            return raw_ctx

        handler = {
            IntentType.JSON_DIEM_CHUAN      : self._ctx_diem_chuan,
            IntentType.JSON_HOC_PHI         : self._ctx_hoc_phi,
            IntentType.JSON_CHI_TIEU_TO_HOP : self._ctx_chi_tieu_to_hop,
            IntentType.JSON_QUY_DOI_DIEM    : self._ctx_quy_doi_diem,
            IntentType.JSON_DAU_TRUOT       : self._ctx_dau_truot,
            IntentType.RAG_MO_TA_NGANH      : self._ctx_rag,
            IntentType.RAG_FAQ              : self._ctx_rag,
            IntentType.RAG_TRUONG_HOC_BONG  : self._ctx_rag,
            IntentType.UNKNOWN              : self._ctx_rag,
            IntentType.GREETING             : self._ctx_rag,
            IntentType.OFF_TOPIC            : self._ctx_rag,
        }
        raw_ctx = handler.get(intent.intent_type, self._ctx_rag)(query, intent, tracker)

        if raw_ctx not in ("__GREETING__", "__OFF_TOPIC__", "__CANCELLED__") and "__CLARIFY__" not in raw_ctx:
            raw_ctx = self._compressor.compress(raw_ctx, query=query)

        return raw_ctx

    def _ctx_rag(self, query, intent, tracker) -> str:
        result: RetrievalResult = self._retriever.retrieve_v2(query, intent_type=intent.intent_type)
        self._last_retrieval = result

        if DEBUG_MODE:
            logger.info(f"[RAG] tier={result.tier} miss={result.is_miss} "
                        f"rewritten='{result.rewritten_query[:60]}' "
                        f"hyde={result.hyde_used} chunks={len(result.chunks)}")

        if result.is_miss:
            return self._retriever._build_miss_context(result)

        parts = []
        for chunk in result.chunks:
            header   = f"[{chunk.source} — {chunk.section}]" if chunk.section else f"[{chunk.source}]"
            tier_tag = f" [tier={result.tier}]" if result.tier > 1 else ""
            parts.append(f"{header}{tier_tag}\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    # JSON handlers giữ nguyên từ v5 ──────────────────────────────────────────

    def _ctx_diem_chuan(self, query, intent, tracker) -> str:
        e   = intent.entities
        ten = e.get("nganh") or tracker.get("nganh") or query
        nam = e.get("nam")
        pt  = e.get("phuong_thuc")
        q_lower = query.lower()
        SO_SANH_KW = ["so sánh", "xu hướng", "qua các năm", "nhiều năm", "lịch sử", "3 năm", "các năm"]
        if any(kw in q_lower for kw in SO_SANH_KW):
            result = get_lich_su_diem_chuan(ten)
        elif nam:
            result = get_diem_chuan(ten, nam=nam, phuong_thuc=pt)
        else:
            result = get_diem_chuan_moi_nhat(ten)
        ctx = fmt_diem_chuan(result)
        if not result["found"]:
            rag_result = self._retriever.retrieve_v2(query)
            self._last_retrieval = rag_result
            ctx += "\n\n" + self._retriever.retrieve_as_context(query)
        return ctx

    def _ctx_hoc_phi(self, query, intent, tracker) -> str:
        nganh  = intent.entities.get("nganh") or tracker.get("nganh", "")
        result = get_hoc_phi(nganh) if nganh else {"found": False}
        if not result.get("found"): result = get_hoc_phi(query)
        if not result.get("found"): result = get_hoc_phi("")
        return fmt_hoc_phi(result)

    def _ctx_chi_tieu_to_hop(self, query, intent, tracker) -> str:
        e        = intent.entities
        contexts = []
        if e.get("to_hop"):
            contexts.append(fmt_nganh_theo_to_hop(get_nganh_theo_to_hop(e["to_hop"])))
        nganh = e.get("nganh") or tracker.get("nganh")
        if nganh:
            contexts.append(fmt_chi_tieu_nganh(get_chi_tieu_nganh(nganh)))
        if "2026" in query or "năm tới" in query.lower():
            contexts.append(fmt_chi_tieu_2026(get_chi_tieu_tong_2026()))
        if not contexts:
            rag_result = self._retriever.retrieve_v2(query)
            self._last_retrieval = rag_result
            contexts.append(self._retriever.retrieve_as_context(query))
        return "\n\n".join(contexts)

    def _ctx_quy_doi_diem(self, query, intent, tracker) -> str:
        e        = intent.entities
        contexts = []
        q_lower  = query.lower()
        diem     = e.get("diem")
        diem_30  = diem
        if diem:
            if "hsa" in q_lower or "năng lực" in q_lower:
                r = quy_doi_HSA(diem)
                contexts.append(fmt_quy_doi(r))
                if r["found"]: diem_30 = r["diem_quy_doi"]
            elif "tsa" in q_lower or "tư duy" in q_lower:
                r = quy_doi_TSA(diem)
                contexts.append(fmt_quy_doi(r))
                if r["found"]: diem_30 = r["diem_quy_doi"]
            elif "học bạ" in q_lower:
                r = quy_doi_KQHB(diem)
                contexts.append(fmt_quy_doi(r))
                if r["found"]: diem_30 = round(r["diem_quy_doi"] * 3, 2)
        kv = e.get("khu_vuc") or tracker.get("khu_vuc")
        dt = e.get("doi_tuong") or tracker.get("doi_tuong")
        if kv and diem_30:
            contexts.append(fmt_tinh_diem_uu_tien(tinh_diem_uu_tien(diem_30, kv, dt)))
        elif kv:
            r = get_diem_uu_tien_khu_vuc(kv)
            if r["found"]:
                contexts.append(f"Mức điểm ưu tiên tối đa {r['ten']} ({r['ma']}): +{r['diem']} điểm\nLưu ý: Nếu tổng điểm ≥ 22.5, điểm ưu tiên thực tế sẽ thấp hơn.")
        if not contexts:
            rag_result = self._retriever.retrieve_v2(query)
            self._last_retrieval = rag_result
            contexts.append(self._retriever.retrieve_as_context(query))
        return "\n\n".join(contexts)

    _KHOA_KEYWORDS = [
        # Tên đầy đủ các trường — thêm mới
        "trường cơ khí - ô tô", "trường cơ khí ô tô", "cơ khí - ô tô",
        "trường cntt và truyền thông", "trường cntt & truyền thông",
        "trường công nghệ thông tin và truyền thông",
        "trường kinh tế", "trường ngoại ngữ", "trường ngoại ngữ du lịch",
        "trường ngoại ngữ - du lịch", "trường điện", "trường điện - điện tử",
        "khoa công nghệ hóa", "khoa hóa", "khoa dệt may", "khoa may",
        "khoa công nghệ may",
        # Tên ngắn — giữ từ v1
        "cntt", "công nghệ thông tin", "cơ khí", "kinh tế",
        "du lịch", "dệt may", "ngoại ngữ", "ngôn ngữ", "thực phẩm",
        "truyền thông", "hóa", "điện điện tử", "ô tô",
    ]
    _KHOA_QUERY_PATTERNS = [
        r"ngành nào.*(khoa|trường|trong)\s+(.+?)(?:\?|$)",
        r"(khoa|trường)\s+(.+?)\s+(?:có|gồm|những)\s+ngành",
        r"đậu.*(khoa|trường)\s+(.+?)(?:\?|$)",
        r"vào.*(khoa|trường)\s+(.+?)(?:\?|$)",
        r"(khoa|trường)\s+(.+?)\s+(?:điểm chuẩn|xét tuyển)",
    ]
    _STRIP_TRAILING = ["có", "gồm", "gồm những", "và", "với", "thì", "là"]

    def _detect_khoa(self, query: str) -> Optional[str]:
        """
        [FIX v7] Phát hiện câu hỏi về trường/khoa từ query.

        Thứ tự ưu tiên:
        1. Dùng _extract_truong_khoa_from_query (pattern mạnh, bắt tên đầy đủ)
        2. Verify bằng _resolve_khoa để đảm bảo là tên hợp lệ
        3. Fallback: keyword search như cũ
        """
        # Bước 1: Thử extract tên trường/khoa từ câu hỏi
        extracted = _extract_truong_khoa_from_query(query)
        if extracted:
            # Bước 2: Verify — nếu là tên hợp lệ thì dùng
            if _NGANH_V2_AVAILABLE:
                resolved = _resolve_khoa(extracted)
                if resolved is not None:
                    return extracted
            else:
                # Không có v2, dùng luôn nếu đủ dài
                if len(extracted) >= 3:
                    return extracted

        # Bước 3: Fallback keyword search (giữ logic cũ)
        q = query.lower()
        for kw in sorted(self._KHOA_KEYWORDS, key=len, reverse=True):
            if kw in q:
                return kw
        for pat in self._KHOA_QUERY_PATTERNS:
            m = re.search(pat, q)
            if m:
                khoa_raw = m.group(m.lastindex).strip().rstrip("?.,!")
                for word in self._STRIP_TRAILING:
                    if khoa_raw.endswith(" " + word):
                        khoa_raw = khoa_raw[:-(len(word) + 1)].strip()
                if len(khoa_raw) > 2:
                    return khoa_raw
        return None

    def _ctx_dau_truot(self, query, intent, tracker) -> str:
        e     = intent.entities
        nganh = e.get("nganh") or tracker.get("nganh")
        diem  = e.get("diem")  or tracker.get("diem")
        kv    = e.get("khu_vuc") or tracker.get("khu_vuc")
        dt    = e.get("doi_tuong") or tracker.get("doi_tuong")
        q_lower = query.lower()

        if self._pending.get("waiting_for") == "phuong_thuc":
            if self._is_cancel_text(query):
                self._pending = {}
                return "__CANCELLED__"
            if nganh is None: nganh = self._pending.get("nganh")
            if diem  is None: diem  = self._pending.get("diem")
            if kv    is None: kv    = self._pending.get("khu_vuc")
            self._pending = {}

        khoa = self._detect_khoa(query)
        if khoa and diem and not nganh:
            return self._ctx_dau_truot_theo_khoa(query, diem, khoa, kv, dt)

        if not nganh or not diem:
            if khoa and not diem:
                dc_khoa  = get_diem_chuan_theo_khoa(khoa)
                ctx_khoa = fmt_diem_chuan_theo_khoa(dc_khoa)
                return ctx_khoa + "\n\nLưu ý: Bạn cho tôi biết điểm của bạn để tôi so sánh cụ thể hơn nhé."
            rag_ctx = self._retriever.retrieve_as_context(query)
            return (rag_ctx + "\n\nLưu ý: Để tham khảo điểm xét tuyển, tôi cần biết:\n"
                    "  - Tên ngành bạn muốn xét tuyển\n  - Tổng điểm 3 môn (thang 30) hoặc điểm TSA/HSA\n"
                    "  - Khu vực (KV1/KV2/KV2-NT/KV3) nếu có")

        contexts = []
        diem_30  = diem
        pt_filter = None
        if "tư duy" in q_lower or "tsa" in q_lower:
            r = quy_doi_TSA(diem)
            if r["found"]: diem_30 = r["diem_quy_doi"]; pt_filter = "PT5"; contexts.append(fmt_quy_doi(r))
        elif "năng lực" in q_lower or "hsa" in q_lower:
            r = quy_doi_HSA(diem)
            if r["found"]: diem_30 = r["diem_quy_doi"]; pt_filter = "PT4"; contexts.append(fmt_quy_doi(r))
        elif "học bạ" in q_lower or "pt2" in q_lower:
            r = quy_doi_KQHB(diem)
            if r["found"]: diem_30 = round(r["diem_quy_doi"] * 3, 2); pt_filter = "PT2"; contexts.append(fmt_quy_doi(r) + f"\n  → Điểm 3 môn thang 30: {diem_30}")
        elif any(kw in q_lower for kw in ["thpt", "thi thpt", "pt3", "tốt nghiệp"]):
            pt_filter = "PT3"
        else:
            self._pending = {"waiting_for": "phuong_thuc", "nganh": nganh, "diem": diem, "khu_vuc": kv}
            dc_all = get_diem_chuan(nganh, nam=2025)
            ctx_dc = "\n\nĐiểm chuẩn tham khảo năm 2025:\n" + fmt_diem_chuan(dc_all) if dc_all["found"] else ""
            return (f"Bạn có {diem} điểm muốn xét tuyển ngành {nganh}.{ctx_dc}\n\n"
                    f"__CLARIFY__ Bạn thi theo phương thức nào?\n1. Thi THPT (PT3)\n2. Đánh giá tư duy TSA (PT5)\n"
                    f"3. Đánh giá năng lực HSA (PT4)\n4. Xét học bạ (PT2)")

        diem_xet = diem_30
        if kv:
            ut = tinh_diem_uu_tien(diem_30, kv, dt)
            if ut["found"]:
                diem_xet = round(diem_30 + ut["diem_uu_tien_thuc"], 2)
                contexts.append(fmt_tinh_diem_uu_tien(ut))

        _dc_raw  = get_diem_chuan(nganh, nam=None)
        dc       = _dc_raw["ket_qua"] if _dc_raw["found"] else []
        _r2025   = next((r for r in dc if r["nam"] == 2025 and not r.get("phuong_thuc_code")), None)
        _dc_parts = []
        if _r2025: _dc_parts.append(f"• 2025 — Chung: {_r2025['diem_chuan']} điểm")
        dc_val = _r2025["diem_chuan"] if _r2025 else None

        if _dc_parts: contexts.append(f"Điểm chuẩn ngành {nganh} gần nhất:\n" + "\n".join(_dc_parts))

        if dc_val is not None:
            lines = [f"Điểm xét tuyển của bạn: {diem_xet} điểm (thang 30)"]
            if _r2025:
                chenh = round(diem_xet - _r2025["diem_chuan"], 2)
                lines.append(f"So với điểm chung 2025 ({_r2025['diem_chuan']}): {'cao hơn' if chenh >= 0 else 'thấp hơn'} {abs(chenh)} điểm.")
            lines.append("Điểm chuẩn 2026 chưa được công bố. Số liệu trên chỉ mang tính tham khảo.")
            contexts.append("\n".join(lines))

        contexts.append("Ghi chú: Theo dõi thông báo chính thức tại tuyensinh.haui.edu.vn.")
        return "\n\n".join(ctx for ctx in contexts if ctx)

    def _ctx_nganh_theo_khoa(self, query, khoa) -> str:
        """
        [FIX v7] Dùng nganh_v2 có alias đầy đủ trước, fallback về điểm chuẩn rồi RAG.
        Giải quyết: "Trường Cơ khí - Ô tô có bao nhiêu ngành?" không trả lời được.
        """
        # Bước 1: Thử lấy danh sách ngành (dùng v2 nếu có, v1 nếu không)
        if _NGANH_V2_AVAILABLE:
            result_nganh = _get_nganh_theo_khoa_v2(khoa)
        else:
            result_nganh = get_nganh_theo_khoa(khoa)

        if result_nganh.get("found"):
            # Nếu user hỏi về điểm → bổ sung điểm chuẩn
            q = query.lower()
            if any(w in q for w in ["điểm", "đậu", "trượt", "chuẩn"]):
                dc_khoa = get_diem_chuan_theo_khoa(khoa)
                if dc_khoa.get("found"):
                    return fmt_diem_chuan_theo_khoa(dc_khoa, diem_user=None)
            return fmt_nganh_theo_khoa(result_nganh)

        # Bước 2: Thử điểm chuẩn theo khoa (API hiện tại)
        dc_khoa = get_diem_chuan_theo_khoa(khoa)
        if dc_khoa.get("found"):
            return fmt_diem_chuan_theo_khoa(dc_khoa, diem_user=None)

        # Bước 3: Fallback RAG
        rag_result = self._retriever.retrieve_v2(query)
        self._last_retrieval = rag_result
        return self._retriever.retrieve_as_context(query)

    def _ctx_dau_truot_theo_khoa(self, query, diem, khoa, kv, dt) -> str:
        contexts = []
        diem_30  = float(diem)
        if kv:
            ut = tinh_diem_uu_tien(diem_30, kv, dt)
            if ut["found"]: diem_30 = ut["diem_xet_tuyen"]; contexts.append(fmt_tinh_diem_uu_tien(ut))
        dc_khoa = get_diem_chuan_theo_khoa(khoa)
        if not dc_khoa["found"]:
            rag_result = self._retriever.retrieve_v2(query)
            self._last_retrieval = rag_result
            return self._retriever.retrieve_as_context(query)
        contexts.append(fmt_diem_chuan_theo_khoa(dc_khoa, diem_user=diem_30))
        return "\n\n".join(ctx for ctx in contexts if ctx)

    def reset(self):
        self._pending = {}
        self._last_retrieval = None

    def _is_cancel_text(self, text) -> bool:
        t = text.lower().strip().rstrip("!.?")
        CANCEL = {"thôi", "bỏ qua", "không cần", "hủy", "cancel", "bỏ đi", "không"}
        return t in CANCEL or any(p in t for p in CANCEL)

    def _ctx_co_cau(self, query: str) -> str:
        """
        [FIX v7] Trả về context về cơ cấu tổ chức HaUI từ structured data.
        Dùng khi user hỏi "HaUI có bao nhiêu trường/khoa?", "trường trực thuộc là gì?"
        """
        if _NGANH_V2_AVAILABLE:
            try:
                co_cau = get_co_cau_truong_khoa()
                if not co_cau.get("found"):
                    return ""
                lines = [
                    "Cơ cấu tổ chức đào tạo Đại học Công nghiệp Hà Nội (HaUI):\n",
                    "5 Trường trực thuộc:"
                ]
                for t in co_cau["truong_truc_thuoc"]:
                    lines.append(f"  - {t['ten']}")
                lines.append("\n4 Khoa trực thuộc:")
                for k in co_cau["khoa_truc_thuoc"]:
                    lines.append(f"  - {k['ten']}")
                lines.append(f"\n{co_cau['ghi_chu']}")
                return "\n".join(lines)
            except Exception as e:
                logger.debug(f"_ctx_co_cau error: {e}")
                return ""
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# CHATBOT v6
# ══════════════════════════════════════════════════════════════════════════════

class Chatbot:
    """
    Chatbot tư vấn tuyển sinh HaUI v6 — Kiến trúc thông minh.

    Thay đổi cốt lõi:
      [A] Router v3  → LLM classify intent, không cần pattern
      [B] Retriever v5 → Query Rewriting + HyDE + Self-reflection
      [C] System prompt → Nhóm A/B, linh hoạt hơn với thông tin mô tả
    """

    _CANCEL_PHRASES = {
        "thôi", "bỏ qua", "không cần", "không muốn", "dừng lại",
        "cancel", "hủy", "không hỏi nữa", "bỏ đi", "thôi được",
        "không", "thôi không", "bỏ", "đủ rồi", "không cần nữa",
    }

    _OFF_TOPIC_REPLY = (
        "Xin lỗi, tôi chỉ có thể tư vấn về tuyển sinh HaUI — "
        "điểm chuẩn, học phí, ngành học, phương thức xét tuyển và các thông tin liên quan.\n\n"
        "Bạn có câu hỏi nào về tuyển sinh không?"
    )

    _GREETING_REPLIES = [
        "Xin chào! Tôi là trợ lý tư vấn tuyển sinh HaUI.\nBạn cần tư vấn gì về tuyển sinh không?",
        "Chào bạn! Tôi có thể giúp bạn tra cứu điểm chuẩn, học phí, ngành học.\nBạn muốn hỏi gì?",
        "Xin chào! Rất vui được hỗ trợ bạn.\nBạn đang quan tâm đến ngành nào tại HaUI?",
    ]
    _THANKS_REPLIES = [
        "Không có gì! Nếu cần thêm thông tin về tuyển sinh HaUI, tôi luôn sẵn sàng.",
        "Rất vui được giúp bạn! Hãy hỏi thêm nếu cần nhé.",
    ]
    _BYE_REPLIES = [
        "Tạm biệt! Chúc bạn thi tốt và đạt kết quả như mong muốn!",
        "Hẹn gặp lại! Chúc bạn may mắn trong kỳ xét tuyển!",
    ]

    def __init__(self, model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL,
                 retriever=None, use_hybrid=True):
        # LLM cho generate answer
        if USE_GROQ:
            logger.info("Backend: Groq API")
            self._llm = GroqLLM()
        else:
            logger.info("Backend: Ollama local")
            self._llm = OllamaLLM(model=model, base_url=base_url)

        # Retriever v5
        self._retriever = retriever or (
            Retriever(use_reranker=True, use_bm25=True)
            if use_hybrid else
            Retriever(use_reranker=True, use_bm25=False)
        )

        # Router v3 — LLM-based
        self._router = Router()
        self._router.init_llm(base_url=base_url, model=model)
        try:
            self._router.init_embedder(self._retriever._embedder.embed_query)
            logger.info("Router: embedding fallback ready")
        except Exception as e:
            logger.warning(f"Router embedding fallback failed: {e}")

        # Context builder, history, tracking
        self._ctx_builder = ContextBuilder(self._retriever)
        self._history_mgr = HistoryManager(max_pairs=MAX_HISTORY)
        self._tracker     = EntityTracker()
        self._rewriter    = ConversationQueryRewriter()
        self._pending     : dict = {}

        # Cache + Hedger
        self._cache  = ResponseCache()
        self._hedger = ConfidenceHedger()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_cancel(self, text: str) -> bool:
        t = text.lower().strip().rstrip("!.?")
        return t in self._CANCEL_PHRASES or any(p in t for p in self._CANCEL_PHRASES)

    def _pick_greeting_reply(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["cảm ơn", "thanks", "thank", "cám ơn"]):
            return random.choice(self._THANKS_REPLIES)
        if any(w in q for w in ["bye", "tạm biệt", "hẹn gặp"]):
            return random.choice(self._BYE_REPLIES)
        return random.choice(self._GREETING_REPLIES)

    def _build_user_prompt_patched(self, user_message: str, context: str) -> str:
        """
    Thêm nhắc nhở ngay trong user prompt để LLM không quên nguyên tắc.
    Đặc biệt quan trọng khi context có [RETRIEVAL_MISS] — LLM phải dừng lại.
        """
    # Nếu context báo miss → nhắc LLM rõ ràng không được bịa
        if "[RETRIEVAL_MISS" in context or "[MISS]" in context:
            return (
                f"[THÔNG TIN THAM KHẢO]\n{context}\n\n"
                f"[LƯU Ý QUAN TRỌNG] Context trên báo hiệu không tìm thấy thông tin. "
                f"Hãy trả lời BẰNG TIẾNG VIỆT theo đúng mẫu: "
                f"'Tôi chưa có thông tin về X. Vui lòng xem tuyensinh.haui.edu.vn.'\n"
                f"TUYỆT ĐỐI không trả lời bằng tiếng Trung hay ngôn ngữ khác.\n\n"
                f"[CÂU HỎI]\n{user_message}"
            )
        return (
            f"[THÔNG TIN THAM KHẢO]\n{context}\n\n"
            f"[YÊU CẦU] Trả lời BẰNG TIẾNG VIỆT, chỉ dựa vào thông tin tham khảo trên.\n\n"
            f"[CÂU HỎI]\n{user_message}"
        )

    def _build_system_with_summary(self) -> str:
        return SYSTEM_PROMPT + self._history_mgr.get_summary_prefix()

    def _try_fast_path(self, query: str):
        from src.pipeline.router import _fast_path
        quick = _fast_path(query)
        if quick and quick[0] in (IntentType.GREETING, IntentType.OFF_TOPIC) and quick[1] >= 0.65:
            intent  = Intent(intent_type=quick[0], confidence=quick[1], entities={}, method="rule")
            context = self._ctx_builder.build(query, intent, self._tracker)
            return intent, context
        return None

    def _get_retrieval_meta(self) -> tuple[int, bool, str]:
        r = self._ctx_builder._last_retrieval
        if r is None: return 1, False, ""
        return r.tier, r.is_miss, r.miss_reason

    _MAX_INPUT_CHARS = 500
    _MIN_INPUT_CHARS = 2
    _SPAM_PATTERN    = re.compile(r'(.)\1{15,}|[\x00-\x08\x0b-\x1f\x7f]')

    def _validate_input(self, message: str) -> str | None:
        stripped = message.strip()
        if len(stripped) < self._MIN_INPUT_CHARS: return ""
        if len(stripped) > self._MAX_INPUT_CHARS:
            return f"Câu hỏi quá dài ({len(stripped)} ký tự). Vui lòng rút gọn dưới {self._MAX_INPUT_CHARS} ký tự."
        if self._SPAM_PATTERN.search(stripped):
            return "Tôi không hiểu câu hỏi này. Bạn có thể đặt câu hỏi rõ hơn không?"
        return None

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> ChatResponse:
        p = LatencyProfiler()
        p.mark("start")

        user_message = _light_normalize(user_message)
        err = self._validate_input(user_message)
        if err is not None:
            if err: self._history_mgr.add(user_message, err)
            return ChatResponse(answer=err, intent=IntentType.UNKNOWN, method="rule", context="", confidence=1.0)

        if self._pending and self._is_cancel(user_message):
            self._pending = {}
            answer = "Được rồi, tôi đã hủy. Bạn có muốn hỏi gì khác không?"
            self._history_mgr.add(user_message, answer)
            return ChatResponse(answer=answer, intent=IntentType.UNKNOWN, method="rule", context="", confidence=1.0)

        fast = self._try_fast_path(user_message)
        if fast:
            intent, context = fast
            answer = self._pick_greeting_reply(user_message) if context == "__GREETING__" else self._OFF_TOPIC_REPLY
            self._history_mgr.add(user_message, answer)
            p.mark("fast_done"); p.report(query=user_message)
            return ChatResponse(answer=answer, intent=intent.intent_type, method=intent.method, context=context, confidence=intent.confidence)

        recent    = self._history_mgr.get_recent(4)
        rewritten = self._rewriter.rewrite(user_message, self._tracker, recent)
        effective_query = rewritten
        p.mark("rewrite")

        # [A] Router v3 — LLM classify
        intent = self._router.classify(effective_query)
        p.mark("classify")
        if DEBUG_MODE:
            logger.info(f"[Router] method={intent.method} intent={intent.intent_type} conf={intent.confidence:.2f}")

        # [FIX v7] Dùng entity extractor v2 để nhận dạng tên trường/khoa
        raw_entities = _extract_entities_v2(effective_query)
        for k, v in raw_entities.items():
            intent.entities.setdefault(k, v)
        intent.entities = self._tracker.resolve(effective_query, intent.entities)

        cached_answer = self._cache.get(effective_query, intent.intent_type, intent.entities)
        if cached_answer:
            p.mark("cache_hit"); p.report(query=user_message)
            self._history_mgr.add(user_message, cached_answer)
            return ChatResponse(answer=cached_answer, intent=intent.intent_type, method=intent.method,
                                context="[cached]", confidence=intent.confidence,
                                rewritten_query=rewritten if rewritten != user_message else "",
                                cache_hit=True, retrieval_tier=0, router_method=intent.method)

        # [B] Retriever v5 — Query Rewriting + HyDE + Self-reflection
        self._ctx_builder._last_retrieval = None
        context = self._ctx_builder.build(effective_query, intent, self._tracker)
        p.mark("context")

        self._tracker.update(intent.entities)

        if context == "__CANCELLED__":
            answer = "Được rồi, tôi đã hủy. Bạn có muốn hỏi gì khác không?"
        elif context == "__GREETING__":
            answer = self._pick_greeting_reply(user_message)
        elif context == "__OFF_TOPIC__":
            answer = self._OFF_TOPIC_REPLY
        else:
            if "__CLARIFY__" in context:
                context = context.replace("__CLARIFY__", "")
            user_prompt   = self._build_user_prompt_patched(user_message, context)
            system_prompt = self._build_system_with_summary()
            answer = self._llm.generate(system=system_prompt, history=self._history_mgr.get_messages(), user_msg=user_prompt)

            tier, is_miss, miss_reason = self._get_retrieval_meta()
            answer = self._hedger.apply(answer, intent.intent_type, tier, is_miss, miss_reason, intent.confidence)
            self._cache.set(effective_query, intent.intent_type, intent.entities, answer)

        p.mark("generate"); p.report(query=user_message)
        self._history_mgr.add(user_message, answer)
        tier, is_miss, _ = self._get_retrieval_meta()
        last_r = self._ctx_builder._last_retrieval

        return ChatResponse(
            answer=answer, intent=intent.intent_type, method=intent.method,
            context=context, confidence=intent.confidence,
            rewritten_query=rewritten if rewritten != user_message else "",
            retrieval_tier=tier,
            hyde_used=last_r.hyde_used if last_r else False,
            router_method=intent.method,
        )

    def chat_stream(self, user_message: str) -> Iterator[str]:
        p = LatencyProfiler()
        p.mark("start")

        user_message = _light_normalize(user_message)
        err = self._validate_input(user_message)
        if err is not None:
            if err: self._history_mgr.add(user_message, err); yield err
            return

        fast = self._try_fast_path(user_message)
        if fast:
            intent, context = fast
            answer = self._pick_greeting_reply(user_message) if context == "__GREETING__" else self._OFF_TOPIC_REPLY
            self._history_mgr.add(user_message, answer)
            p.mark("fast_done"); p.report(query=user_message)
            yield answer; return

        if self._pending and self._is_cancel(user_message):
            self._pending = {}
            answer = "Được rồi, tôi đã hủy câu hỏi. Bạn có muốn hỏi gì khác không?"
            self._history_mgr.add(user_message, answer); yield answer; return

        recent    = self._history_mgr.get_recent(4)
        rewritten = self._rewriter.rewrite(user_message, self._tracker, recent)
        effective_query = rewritten
        p.mark("rewrite")

        intent = self._router.classify(effective_query)
        p.mark("classify")

        # [FIX v7] Dùng entity extractor v2
        raw_entities = _extract_entities_v2(effective_query)
        for k, v in raw_entities.items():
            intent.entities.setdefault(k, v)
        intent.entities = self._tracker.resolve(effective_query, intent.entities)

        cached_answer = self._cache.get(effective_query, intent.intent_type, intent.entities)
        if cached_answer:
            p.mark("cache_hit"); p.report(query=user_message)
            self._history_mgr.add(user_message, cached_answer)
            yield cached_answer; return

        self._ctx_builder._last_retrieval = None
        context = self._ctx_builder.build(effective_query, intent, self._tracker)
        self._tracker.update(intent.entities)
        p.mark("context")

        if context == "__CANCELLED__":
            answer = "Được rồi, tôi đã hủy câu hỏi. Bạn có muốn hỏi gì khác không?"
            self._history_mgr.add(user_message, answer); yield answer; return

        if "__CLARIFY__" in context:
            context = context.replace("__CLARIFY__", "")

        tier, is_miss, miss_reason = self._get_retrieval_meta()
        needs_hedge  = self._hedger.should_hedge(intent.intent_type, tier, is_miss, intent.confidence)
        hedge_prefix = ""
        hedge_suffix = ""
        if needs_hedge:
            hedge_prefix = self._hedger.build_prefix(tier, is_miss, miss_reason, intent.confidence)
            hedge_suffix = self._hedger.build_suffix(intent.intent_type, tier, is_miss)
            if hedge_prefix: yield hedge_prefix + "\n\n"

        user_prompt   = self._build_user_prompt_patched(user_message, context)
        system_prompt = self._build_system_with_summary()
        buf           = StreamBuffer()
        full_answer   = ""
        first_token   = True

        for raw_token in self._llm.generate_stream(system=system_prompt, history=self._history_mgr.get_messages(), user_msg=user_prompt):
            if first_token:
                p.mark("first_token"); p.report(query=user_message)
                first_token = False
            full_answer += raw_token
            for chunk in buf.feed(raw_token): yield chunk

        for chunk in buf.flush(): yield chunk
        if hedge_suffix: yield hedge_suffix

        complete_answer = (hedge_prefix + "\n\n" if hedge_prefix else "") + full_answer
        if hedge_suffix: complete_answer += hedge_suffix
        self._cache.set(effective_query, intent.intent_type, intent.entities, complete_answer)
        self._history_mgr.add(user_message, complete_answer)

    def reset(self):
        self._history_mgr.clear()
        self._tracker     = EntityTracker()
        self._pending     = {}
        self._ctx_builder.reset()
        self._cache.invalidate()

    def cache_stats(self):
        return self._cache.stats()


# ══════════════════════════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO if not DEBUG_MODE else logging.DEBUG,
                        format="%(levelname)s — %(message)s")

    print("Khởi động HaUI Chatbot v6...")
    try:
        bot = Chatbot()
    except RuntimeError as e:
        print(f"\nLỗi: {e}"); sys.exit(1)

    print("Sẵn sàng! Gõ 'exit' để thoát.\n")
    while True:
        try:
            q = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!"); break
        if not q: continue
        if q.lower() in ("exit", "quit", "thoát"): print("Tạm biệt!"); break

        print("Bot: ", end="", flush=True)
        for token in bot.chat_stream(q):
            print(token, end="", flush=True)
        print("\n")