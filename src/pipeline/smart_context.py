"""
smart_context.py — Cross-source context builder

Giải quyết vấn đề:
  P3: RAG và JSON không có cross-fallback
  P4: Context thiếu / sai → LLM vẫn hallucinate

Pipeline:
  1. Cố gắng JSON lookup (chính xác, không hallucinate)
  2. Nếu JSON miss → thử RAG
  3. Nếu RAG tier >= 2 → thêm JSON context bổ sung
  4. ConfidenceScorer đánh giá context trước khi đưa cho LLM
  5. Nếu confidence thấp → inject cảnh báo vào system prompt
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("haui.smart_context")


@dataclass
class ContextResult:
    text        : str
    source      : str       # "json", "rag", "json+rag", "miss"
    confidence  : float     # 0.0 - 1.0
    should_hedge: bool      # LLM có nên thêm disclaimer không
    debug_info  : dict


class AnswerConfidenceScorer:
    """
    Đánh giá context trước khi đưa cho LLM.
    
    Mục đích: ngăn LLM hallucinate khi context thiếu/sai.
    Score cao = context đáng tin, LLM có thể trả lời trực tiếp.
    Score thấp = context nghi ngờ, LLM phải thêm disclaimer.
    """

    # Các pattern cho thấy context có số liệu thật
    _NUMERIC_PATTERNS = [
        re.compile(r'\b\d{2}[.,]\d{1,2}\b'),          # điểm chuẩn: 24.5
        re.compile(r'\b\d{1,3}[.,]\d{3}\s*(đồng|vnd)', re.I),  # học phí
        re.compile(r'\b\d+\s*tín\s*chỉ\b', re.I),
        re.compile(r'\b(202[3-9])\b'),                 # năm
        re.compile(r'\bPT[1-5]\b'),                    # phương thức
    ]

    # Pattern cho thấy context là miss/lỗi
    _MISS_PATTERNS = [
        re.compile(r'\[RETRIEVAL_MISS', re.I),
        re.compile(r'không tìm thấy.*ngành', re.I),
        re.compile(r'chưa có thông tin', re.I),
    ]

    def score(self, context: str, query: str, source: str) -> float:
        """
        Returns confidence score 0.0-1.0.
        
        1.0 = JSON exact match (luôn tin tưởng)
        0.8 = RAG tier 1 với nhiều số liệu
        0.5 = RAG tier 2 hoặc context chung
        0.2 = Miss hoặc context không liên quan
        """
        if not context or len(context) < 20:
            return 0.1

        # Miss patterns → score rất thấp
        for p in self._MISS_PATTERNS:
            if p.search(context):
                return 0.2

        # JSON source → luôn high confidence
        if source == "json":
            return 0.95

        # Đếm số liệu thực trong context
        numeric_hits = sum(1 for p in self._NUMERIC_PATTERNS if p.search(context))

        # Đếm từ khoá query xuất hiện trong context
        query_words = {w for w in re.findall(r'\w{3,}', query.lower())}
        context_lower = context.lower()
        kw_overlap = sum(1 for w in query_words if w in context_lower)
        kw_ratio   = kw_overlap / max(len(query_words), 1)

        base = 0.5
        if numeric_hits >= 3:
            base += 0.2
        elif numeric_hits >= 1:
            base += 0.1

        if kw_ratio >= 0.6:
            base += 0.2
        elif kw_ratio >= 0.3:
            base += 0.1

        return min(base, 0.9)


class SmartContextBuilder:
    """
    Xây dựng context theo chuỗi fallback:
    JSON first → RAG → Cross-source merge

    Mỗi bước có confidence scoring.
    Nếu confidence thấp, inject warning để LLM không hallucinate.
    """

    # Ngưỡng confidence để hedge
    HEDGE_THRESHOLD = 0.55

    def __init__(self, retriever, json_handlers: dict):
        """
        json_handlers: dict mapping intent → callable(query, entities, tracker)
        Ví dụ:
          {
            "diem_chuan": lambda q, e, t: fmt_diem_chuan(get_diem_chuan(e.get("nganh", q))),
            "hoc_phi":    lambda q, e, t: fmt_hoc_phi(get_hoc_phi(e.get("nganh", ""))),
          }
        """
        self._retriever   = retriever
        self._json_hdl    = json_handlers
        self._scorer      = AnswerConfidenceScorer()

    def build(self, query: str, intent_type, entities: dict, tracker) -> ContextResult:
        """
        Trả ContextResult với context tốt nhất có thể.
        """
        from src.pipeline.router import IntentType, JSON_INTENTS

        # ── 1. Thử JSON first cho JSON intents ────────────────────────────
        if intent_type in JSON_INTENTS:
            json_ctx = self._try_json(query, intent_type, entities, tracker)
            if json_ctx and len(json_ctx) > 30 and "Không tìm thấy" not in json_ctx[:50]:
                score = self._scorer.score(json_ctx, query, "json")
                return ContextResult(
                    text=json_ctx, source="json",
                    confidence=score, should_hedge=(score < self.HEDGE_THRESHOLD),
                    debug_info={"intent": intent_type.value, "path": "json_direct"}
                )
            # JSON miss → thử RAG bổ sung
            rag_ctx  = self._try_rag(query, intent_type)
            combined = self._merge(json_ctx or "", rag_ctx)
            score    = self._scorer.score(combined, query, "json+rag")
            return ContextResult(
                text=combined, source="json+rag",
                confidence=score, should_hedge=(score < self.HEDGE_THRESHOLD),
                debug_info={"intent": intent_type.value, "path": "json_miss+rag"}
            )

        # ── 2. RAG first cho RAG intents ──────────────────────────────────
        rag_result = self._retriever.retrieve_v2(query, intent_type=intent_type)
        rag_ctx    = self._format_rag(rag_result)
        rag_score  = self._scorer.score(rag_ctx, query, "rag")

        # RAG tier >= 2 (kém tin cậy) → bổ sung JSON context liên quan
        if rag_result.tier >= 2 or rag_score < 0.55:
            json_supplement = self._try_json_supplement(query, entities, tracker)
            if json_supplement:
                combined = self._merge(rag_ctx, json_supplement)
                score    = self._scorer.score(combined, query, "rag+json")
                return ContextResult(
                    text=combined, source="rag+json",
                    confidence=score, should_hedge=(score < self.HEDGE_THRESHOLD),
                    debug_info={"tier": rag_result.tier, "path": "rag_weak+json_supplement"}
                )

        return ContextResult(
            text=rag_ctx, source="rag",
            confidence=rag_score, should_hedge=(rag_score < self.HEDGE_THRESHOLD),
            debug_info={"tier": rag_result.tier, "hyde": rag_result.hyde_used}
        )

    def _try_json(self, query, intent_type, entities, tracker) -> str | None:
        """Thử gọi JSON handler phù hợp."""
        from src.pipeline.router import IntentType
        key_map = {
            IntentType.JSON_DIEM_CHUAN      : "diem_chuan",
            IntentType.JSON_HOC_PHI         : "hoc_phi",
            IntentType.JSON_CHI_TIEU_TO_HOP : "chi_tieu",
            IntentType.JSON_QUY_DOI_DIEM    : "quy_doi",
            IntentType.JSON_DAU_TRUOT       : "dau_truot",
        }
        key = key_map.get(intent_type)
        if not key or key not in self._json_hdl:
            return None
        try:
            return self._json_hdl[key](query, entities, tracker)
        except Exception as e:
            logger.debug(f"JSON handler '{key}' failed: {e}")
            return None

    def _try_rag(self, query, intent_type) -> str:
        """Thử RAG retrieval."""
        try:
            return self._retriever.retrieve_as_context(query, intent_type=intent_type)
        except Exception as e:
            logger.debug(f"RAG failed: {e}")
            return ""

    def _try_json_supplement(self, query: str, entities: dict, tracker) -> str | None:
        """
        Tìm JSON data bổ sung dựa trên entities trong query.
        Dùng khi RAG không đủ chắc.
        """
        from src.query_json import (
            get_diem_chuan_moi_nhat, fmt_diem_chuan,
            get_hoc_phi, fmt_hoc_phi,
        )
        q = query.lower()
        parts = []

        nganh = entities.get("nganh") or tracker.get("nganh") if tracker else None

        if nganh and any(w in q for w in ["điểm chuẩn", "điểm đầu vào", "đậu", "trượt"]):
            r = get_diem_chuan_moi_nhat(nganh)
            if r.get("found"):
                parts.append(fmt_diem_chuan(r))

        if any(w in q for w in ["học phí", "tiền học"]):
            r = get_hoc_phi(nganh or q)
            if r.get("found"):
                parts.append(fmt_hoc_phi(r))

        return "\n\n".join(parts) if parts else None

    def _format_rag(self, result) -> str:
        """Format RAG result thành string."""
        if result.is_miss:
            return self._retriever._build_miss_context(result)
        parts = []
        for chunk in result.chunks:
            header   = f"[{chunk.source} — {chunk.section}]" if chunk.section else f"[{chunk.source}]"
            tier_tag = f" [tier={result.tier}]" if result.tier > 1 else ""
            parts.append(f"{header}{tier_tag}\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    def _merge(self, primary: str, secondary: str) -> str:
        """Merge hai context, tránh duplicate."""
        if not primary:
            return secondary
        if not secondary:
            return primary
        # Không nối nếu secondary đã có trong primary
        if secondary[:60] in primary:
            return primary
        return primary + "\n\n---\n\n" + secondary

    def build_system_prompt_suffix(self, ctx_result: ContextResult) -> str:
        """
        Thêm vào system prompt dựa trên confidence.
        Nếu confidence thấp → nhắc LLM không được bịa.
        """
        if ctx_result.confidence >= 0.8:
            return ""   # Context tốt, không cần nhắc thêm

        if ctx_result.confidence < 0.3:
            return (
                "\n\n[CẢNH BÁO ĐỘ TIN CẬY THẤP]\n"
                "Context trên có thể không đủ để trả lời chính xác. "
                "Nếu không tìm thấy thông tin rõ ràng trong context, "
                "PHẢI dùng mẫu câu: 'Tôi chưa có thông tin về [X]. "
                "Vui lòng xem tuyensinh.haui.edu.vn hoặc gọi 0243.7655121.'"
            )

        return (
            "\n\n[LƯU Ý]\n"
            "Context trên có thể chưa đầy đủ. "
            "Chỉ trả lời những gì có trong context, không suy đoán thêm."
        )