"""
router.py  (v3 — LLM-based intent classification)

Thay đổi kiến trúc so với v2:

  TRƯỚC (v2):
    Rule-based regex → Embedding similarity → Fallback UNKNOWN
    Vấn đề: rule cứng, không hiểu ngữ nghĩa, dễ miss câu tự nhiên

  SAU (v3):
    Fast rule (greeting/off-topic/JSON rõ ràng) → LLM classify → Fallback embedding
    
  Nguyên tắc:
    1. Fast-path rules   — chỉ dùng cho GREETING, OFF_TOPIC, và JSON intent CÓ số liệu rõ
                           (< 1ms, confidence cao, không cần LLM)
    2. LLM classification — dùng model nhỏ (qwen2.5:3b hoặc cùng model) để hiểu ngữ nghĩa
                           thật sự. Prompt cực ngắn, trả JSON. (~100-200ms)
    3. Embedding fallback — nếu LLM không available, fallback về embedding similarity
    4. UNKNOWN fallback   — cuối cùng, đẩy vào RAG

  Lợi ích:
    - Hiểu "trường thành viên", "cơ cấu đào tạo", "có bao nhiêu khoa" → đúng intent
    - Không cần maintain list pattern/regex
    - Tự động thích nghi với cách diễn đạt mới của user
    - Vẫn fast cho các câu rõ ràng (greeting/off-topic)
"""

from __future__ import annotations

import re
import json
import time
import logging
import requests
import os
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("haui.router")

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11435")
# Dùng model nhỏ hơn để classify nhanh — nếu không có thì dùng model chính
ROUTER_MODEL     = os.getenv("ROUTER_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:14b"))
ROUTER_TIMEOUT   = float(os.getenv("ROUTER_TIMEOUT", "8"))   # giây

# ── Intent types (giữ nguyên để backward compatible với chatbot.py) ───────────

class IntentType(str, Enum):
    JSON_DIEM_CHUAN      = "JSON_DIEM_CHUAN"
    JSON_HOC_PHI         = "JSON_HOC_PHI"
    JSON_CHI_TIEU_TO_HOP = "JSON_CHI_TIEU_TO_HOP"
    JSON_QUY_DOI_DIEM    = "JSON_QUY_DOI_DIEM"
    JSON_DAU_TRUOT       = "JSON_DAU_TRUOT"
    RAG_MO_TA_NGANH      = "RAG_MO_TA_NGANH"
    RAG_FAQ              = "RAG_FAQ"
    RAG_TRUONG_HOC_BONG  = "RAG_TRUONG_HOC_BONG"
    UNKNOWN              = "UNKNOWN"
    OFF_TOPIC            = "OFF_TOPIC"
    GREETING             = "GREETING"

JSON_INTENTS = {
    IntentType.JSON_DIEM_CHUAN,
    IntentType.JSON_HOC_PHI,
    IntentType.JSON_CHI_TIEU_TO_HOP,
    IntentType.JSON_QUY_DOI_DIEM,
    IntentType.JSON_DAU_TRUOT,
}

RAG_INTENTS = {
    IntentType.RAG_MO_TA_NGANH,
    IntentType.RAG_FAQ,
    IntentType.RAG_TRUONG_HOC_BONG,
    IntentType.UNKNOWN,
}

@dataclass
class Intent:
    intent_type : IntentType
    confidence  : float
    entities    : dict = field(default_factory=dict)
    method      : str  = "rule"

    @property
    def is_json(self) -> bool:
        return self.intent_type in JSON_INTENTS

    @property
    def is_rag(self) -> bool:
        return self.intent_type in RAG_INTENTS


# ── LLM Classification Prompt ─────────────────────────────────────────────────

# Prompt cực ngắn, chỉ trả JSON — tối ưu cho latency
_CLASSIFY_SYSTEM = """\
Bạn là bộ phân loại intent cho chatbot tuyển sinh HaUI. Trả lời CHỈ bằng JSON, không giải thích.

INTENT LIST:
- JSON_DIEM_CHUAN: hỏi điểm chuẩn, điểm đầu vào, điểm trúng tuyển
- JSON_HOC_PHI: hỏi học phí, tiền học, chi phí
- JSON_CHI_TIEU_TO_HOP: hỏi tổ hợp môn, chỉ tiêu, phương thức xét tuyển
- JSON_QUY_DOI_DIEM: quy đổi điểm HSA/TSA/học bạ, điểm ưu tiên khu vực
- JSON_DAU_TRUOT: hỏi có đậu không, đủ điểm không, so sánh điểm với chuẩn
- RAG_MO_TA_NGANH: ngành học gì, ra làm gì, cơ hội việc làm, mô tả ngành
- RAG_FAQ: thủ tục đăng ký, hồ sơ, lịch tuyển sinh, hướng dẫn nhập học
- RAG_TRUONG_HOC_BONG: giới thiệu trường, cơ cấu tổ chức, học bổng, ký túc xá, cơ sở vật chất, số lượng khoa/trường/sinh viên
- GREETING: chào hỏi, cảm ơn, tạm biệt
- OFF_TOPIC: không liên quan tuyển sinh

Trả về: {"intent": "TÊN_INTENT", "confidence": 0.0-1.0, "reason": "ngắn gọn"}"""

_CLASSIFY_USER = "Câu hỏi: {query}"


# ── Fast-path rules (chỉ cho các trường hợp CỰC RÕ RÀNG) ────────────────────
# Mục tiêu: giảm LLM call cho greeting/off-topic — không cần thông minh ở đây

_FAST_GREETING = re.compile(
    r"^(xin )?chào|^hello|^hi\b|^hey\b|^ơi\b|^alo\b"
    r"|^cảm ơn|^thanks|^thank|^bye\b|^tạm biệt"
    r"|^ok\b|^okay\b|bạn là ai|cho tôi hỏi",
    re.IGNORECASE,
)

_FAST_OFF_TOPIC = re.compile(
    r"thời tiết|dự báo thời|bóng đá|công thức nấu"
    r"|giá vàng|bitcoin|crypto|chứng khoán"
    r"|viết code.*cho tôi|debug.*lỗi|tôi buồn|tôi vui",
    re.IGNORECASE,
)

# Chỉ fast-path JSON khi CÓ số liệu rõ ràng — tránh miss câu hỏi chung
_FAST_JSON_DIEM = re.compile(
    r"\b\d{2}[.,]\d\b.*(?:điểm|đ)\b"        # "24.5 điểm"
    r"|\bđiểm chuẩn\b"                        # keyword rất rõ
    r"|\bđiểm đầu vào\b",
    re.IGNORECASE,
)

_FAST_JSON_HOC_PHI = re.compile(
    r"\bhọc phí\b"
    r"|\btiền học\b"
    r"|\bmột tín chỉ\b|\bmỗi tín chỉ\b",
    re.IGNORECASE,
)

_FAST_JSON_QUY_DOI = re.compile(
    r"\b(hsa|tsa)\b.*\b\d+\b"               # "HSA 105", "TSA 800"
    r"|\b\d+\b.*(hsa|tsa)\b"
    r"|\bquy đổi\b"
    r"|\bkv[123]\b|\bkhu vực [123]\b",
    re.IGNORECASE,
)

_FAST_OFF_TOPIC = re.compile(
    r"thời tiết|dự báo thời|bóng đá|công thức nấu"
    r"|giá vàng|bitcoin|crypto|chứng khoán"
    r"|viết code.*cho tôi|debug.*lỗi|tôi buồn|tôi vui",
    re.IGNORECASE,
)

_HAUI_CONTEXT = re.compile(
    r"ktx|ký túc xá|phòng.*người|thẻ sinh viên|vietinbank"
    r"|đăng nhập.*hệ thống|mật khẩu|tiếng pháp.*ngôn ngữ"
    r"|ssc\.haui|haui\.edu|cơ sở [12]",
    re.IGNORECASE,
)
def _fast_path(query: str) -> Optional[tuple[IntentType, float]]:
    """
    Chỉ classify các trường hợp CỰC RÕ RÀNG để tránh LLM call không cần thiết.
    Trả None nếu không chắc chắn — để LLM quyết định.
    """
    q = query.strip()

    if _FAST_GREETING.search(q) and len(q.split()) <= 8:
        return IntentType.GREETING, 0.95

    if _FAST_OFF_TOPIC.search(q) and not _HAUI_CONTEXT.search(q):
        return IntentType.OFF_TOPIC, 0.90

    if _FAST_JSON_QUY_DOI.search(q):
        return IntentType.JSON_QUY_DOI_DIEM, 0.88

    if _FAST_JSON_DIEM.search(q):
        return IntentType.JSON_DIEM_CHUAN, 0.88

    if _FAST_JSON_HOC_PHI.search(q):
        return IntentType.JSON_HOC_PHI, 0.88

    return None


# ── Entity extraction (giữ nguyên từ v2 — vẫn cần cho chatbot.py) ─────────────

def _extract_entities_rule(query: str) -> dict:
    q        = query.lower().strip()
    entities = {}

    nam_match = re.search(r"năm (202[3-9]|20[3-9]\d)", q)
    if nam_match:
        entities["nam"] = int(nam_match.group(1))

    pt_match = re.search(r"pt([1-5])", q)
    if pt_match:
        entities["phuong_thuc"] = f"PT{pt_match.group(1)}"

    to_hop_match = re.search(r"\b([abcdx]\d{2}|dd2)\b", q)
    if to_hop_match:
        entities["to_hop"] = to_hop_match.group(1).upper()

    kv_patterns = [
        (r"kv2[-\s]nt\b|khu\s*v[ựu]c\s*2[-\s]nt\b|khu\s*v[ựu]c\s*2\s*n[oô]ng\s*th[oô]n", "KV2-NT"),
        (r"\bkv1\b|khu\s*v[ựu]c\s*1\b", "KV1"),
        (r"\bkv2\b|khu\s*v[ựu]c\s*2\b", "KV2"),
        (r"\bkv3\b|khu\s*v[ựu]c\s*3\b", "KV3"),
    ]
    for _pattern, _kv_value in kv_patterns:
        if re.search(_pattern, q):
            entities["khu_vuc"] = _kv_value
            break

    dt_match = re.search(r"đối tượng 0([1-6])", q)
    if dt_match:
        entities["doi_tuong"] = f"0{dt_match.group(1)}"

    diem_match = re.search(
        r"(?:được|thi|đạt|đạt được|là|scored?)?\s*"
        r"(\d{1,3}[.,]\d{1,2}|\d{3}(?![\d.,])|\d{2}(?![\d.,]))"
        r"\s*(?:điểm|đ\b)?",
        q
    )
    if diem_match:
        entities["diem"] = float(diem_match.group(1).replace(",", "."))

    NGANH_MAP = [
        (r"kỹ thuật phần mềm|\bktpm\b",                       "Kỹ thuật phần mềm"),
        (r"khoa học máy tính|\bksmt\b",                        "Khoa học máy tính"),
        (r"công nghệ thông tin|\bcntt\b",                      "Công nghệ thông tin"),
        (r"robot\b|trí tuệ nhân tạo|robot.*ai|tự động hóa thông minh", "Robot và trí tuệ nhân tạo"),
        (r"an toàn thông tin|an ninh mạng|cyber",              "An toàn thông tin"),
        (r"hệ thống thông tin|\bhttt\b",                       "Hệ thống thông tin"),
        (r"công nghệ đa phương tiện|đa phương tiện|multimedia","Công nghệ đa phương tiện"),
        (r"mạng máy tính|truyền thông dữ liệu",                "Mạng máy tính và truyền thông dữ liệu"),
        (r"công nghệ kỹ thuật máy tính|kỹ thuật máy tính",    "Công nghệ kỹ thuật máy tính"),
        (r"thương mại điện tử|tmđt|tmdt",                      "Thương mại điện tử"),
        (r"điện tử viễn thông|viễn thông",                     "Kỹ thuật điện tử viễn thông"),
        (r"cơ điện tử",                                        "Cơ điện tử"),
        (r"kỹ thuật điện tử|(?<!cơ )(?<!thương mại )điện tử(?! viễn)", "Kỹ thuật điện tử"),
        (r"kỹ thuật điện(?! tử)|(?<!điện )(?<!cơ )điện(?! tử| lạnh)", "Kỹ thuật điện"),
        (r"điện lạnh|kỹ thuật nhiệt|điều hòa",                "Kỹ thuật nhiệt"),
        (r"cơ khí chế tạo|chế tạo máy",                       "Cơ khí chế tạo máy"),
        (r"(?<!\w)cơ khí(?!\s+chế)",                           "Cơ khí"),
        (r"công nghệ ô tô|ô tô\b",                             "Công nghệ kỹ thuật ô tô"),
        (r"kế toán",                                           "Kế toán"),
        (r"kiểm toán",                                         "Kiểm toán"),
        (r"quản trị kinh doanh|qtkd",                          "Quản trị kinh doanh"),
        (r"tài chính|ngân hàng",                               "Tài chính - Ngân hàng"),
        (r"logistics|quản lý chuỗi cung ứng",                  "Logistics và quản lý chuỗi cung ứng"),
        (r"kinh tế số",                                        "Kinh tế số"),
        (r"kinh tế(?! số)",                                    "Kinh tế"),
        (r"công nghệ may|may thời trang|thời trang",           "Công nghệ may"),
        (r"xây dựng|kỹ thuật xây dựng",                       "Kỹ thuật xây dựng"),
        (r"môi trường|kỹ thuật môi trường",                    "Kỹ thuật môi trường"),
        (r"công nghệ thực phẩm|thực phẩm",                    "Công nghệ thực phẩm"),
        (r"du lịch",                                           "Du lịch"),
        (r"ngôn ngữ anh|tiếng anh",                           "Ngôn ngữ Anh"),
        (r"ngôn ngữ trung|tiếng trung",                        "Ngôn ngữ Trung Quốc"),
    ]
    for pattern, ten_chuan in NGANH_MAP:
        if re.search(pattern, q):
            entities["nganh"] = ten_chuan
            break

    return entities


# ── LLM Classifier ────────────────────────────────────────────────────────────

class LLMClassifier:
    """
    Dùng LLM để phân loại intent — hiểu ngữ nghĩa thật sự.
    
    Thiết kế:
    - Prompt cực ngắn (system ~300 tokens) → latency thấp
    - Trả JSON strict → parse không fail
    - Cache kết quả trong session (TTL 5 phút) → không gọi LLM 2 lần cùng câu
    - Timeout ngắn (8s) → fallback nhanh nếu LLM chậm
    """

    _VALID_INTENTS = {e.value for e in IntentType}

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = ROUTER_MODEL):
        self._base_url = base_url.rstrip("/")
        self._model    = model
        self._cache: dict[str, tuple[float, IntentType, float]] = {}
        self._available = self._check_available()

    def _check_available(self) -> bool:
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=3)
            models = [m["name"] for m in resp.json().get("models", [])]
            model_base = self._model.split(":")[0]
            ok = any(model_base in m for m in models)
            if ok:
                logger.info(f"LLMClassifier: {self._model} available")
            else:
                logger.warning(f"LLMClassifier: {self._model} not found, will use embedding fallback")
            return ok
        except Exception as e:
            logger.warning(f"LLMClassifier: Ollama not reachable ({e}), fallback mode")
            return False

    def classify(self, query: str) -> Optional[tuple[IntentType, float]]:
        """
        Trả (IntentType, confidence) hoặc None nếu không available/timeout.
        None → caller sẽ dùng embedding fallback.
        """
        if not self._available:
            return None

        # Cache check
        cache_key = query.lower().strip()
        now = time.monotonic()
        if cache_key in self._cache:
            ts, intent, conf = self._cache[cache_key]
            if now - ts < 300:   # 5 phút TTL
                logger.debug(f"LLMClassifier cache HIT: {intent}")
                return intent, conf

        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": _CLASSIFY_SYSTEM},
                        {"role": "user",   "content": _CLASSIFY_USER.format(query=query)},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.0,    # deterministic
                        "num_predict": 80,     # chỉ cần JSON ngắn
                    },
                },
                timeout=ROUTER_TIMEOUT,
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"].strip()

            # Parse JSON — bỏ markdown fence nếu có
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            # Lấy JSON object đầu tiên trong response
            json_match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
            if not json_match:
                logger.warning(f"LLMClassifier: no JSON in response: {raw[:100]}")
                return None

            data       = json.loads(json_match.group())
            intent_str = data.get("intent", "UNKNOWN").strip()
            confidence = float(data.get("confidence", 0.7))

            # Validate intent value
            if intent_str not in self._VALID_INTENTS:
                logger.warning(f"LLMClassifier: unknown intent '{intent_str}', using UNKNOWN")
                intent_str = "UNKNOWN"

            intent = IntentType(intent_str)

            # Cache
            self._cache[cache_key] = (now, intent, confidence)

            logger.debug(f"LLMClassifier: '{query[:50]}' → {intent} ({confidence:.2f})")
            return intent, confidence

        except requests.exceptions.Timeout:
            logger.warning(f"LLMClassifier: timeout after {ROUTER_TIMEOUT}s, fallback")
            return None
        except Exception as e:
            logger.warning(f"LLMClassifier error: {e}, fallback")
            return None


# ── Embedding Classifier (fallback từ v2 — giữ nguyên) ───────────────────────

INTENT_EXAMPLES: dict[IntentType, list[str]] = {
    IntentType.JSON_DIEM_CHUAN: [
        "Điểm chuẩn ngành CNTT năm 2024 là bao nhiêu?",
        "Năm ngoái vào ngành cơ khí cần bao nhiêu điểm?",
        "Điểm đầu vào ngành kế toán HaUI",
        "Điểm chuẩn các năm của ngành điện tử",
        "Xu hướng điểm chuẩn ngành logistics tăng hay giảm?",
        "Năm 2023 ngành KTPM cần bao nhiêu để đỗ?",
        "Cho tôi xem lịch sử điểm chuẩn ngành quản trị kinh doanh",
        "Cần mấy điểm để vào HaUI ngành CNTT",
        "So sánh điểm chuẩn CNTT 3 năm gần đây",
    ],
    IntentType.JSON_HOC_PHI: [
        "Học phí ngành CNTT bao nhiêu một tín chỉ?",
        "Chi phí học 4 năm tại HaUI khoảng bao nhiêu?",
        "Tiền học mỗi học kỳ ngành kỹ thuật điện là bao nhiêu?",
        "Đóng học phí mỗi năm hết bao nhiêu?",
        "Ngành cơ khí học phí có đắt không?",
        "1 tín chỉ HaUI bao nhiêu tiền",
        "Học phí năm học 2025 2026 HaUI",
    ],
    IntentType.JSON_CHI_TIEU_TO_HOP: [
        "Ngành CNTT xét tuyển tổ hợp môn nào?",
        "Học toán lý hóa có xét tuyển được ngành điện tử không?",
        "Chỉ tiêu tuyển sinh ngành logistics năm nay là bao nhiêu?",
        "Phương thức xét tuyển của HaUI gồm những gì?",
        "Tổ hợp A00 có xét tuyển được ngành nào?",
        "HaUI có mấy phương thức xét tuyển",
    ],
    IntentType.JSON_QUY_DOI_DIEM: [
        "Điểm TSA 850 quy đổi được bao nhiêu điểm?",
        "HSA 105 tương đương mấy điểm thang 30?",
        "Điểm học bạ 8.5 quy đổi thế nào?",
        "90 điểm hsa thì được bao nhiêu điểm",
        "80 tsa khu vực 1 thì điểm xét tuyển là bao nhiêu",
        "Tôi ở khu vực 1 được cộng mấy điểm ưu tiên?",
        "KV2-NT được cộng thêm bao nhiêu điểm?",
    ],
    IntentType.JSON_DAU_TRUOT: [
        "Điểm 24.5 KV1 có đậu ngành CNTT không?",
        "Mình thi được 26 điểm, vào ngành điện tử được không?",
        "Em 23 điểm KV2-NT, trúng tuyển ngành kế toán được không?",
        "Tôi đạt 25 điểm THPT, có trượt ngành CNTT không?",
        "Với điểm HSA 110, em có khả năng đậu ngành robot không?",
    ],
    IntentType.RAG_MO_TA_NGANH: [
        "Ngành robot và trí tuệ nhân tạo sau khi ra trường làm gì?",
        "Học CNTT ở HaUI thì học những môn gì?",
        "Cơ hội việc làm ngành kỹ thuật điện như thế nào?",
        "Ngành logistics có triển vọng không?",
        "So sánh ngành CNTT với kỹ thuật phần mềm khác nhau thế nào?",
        "Lương ngành kế toán sau khi ra trường bao nhiêu?",
        "Nên học ngành nào có nhiều việc làm nhất?",
    ],
    IntentType.RAG_FAQ: [
        "Làm thế nào để đăng ký xét tuyển vào HaUI?",
        "Thủ tục nhập học gồm những gì?",
        "Hạn nộp hồ sơ xét tuyển năm 2025 là khi nào?",
        "Hồ sơ nhập học cần chuẩn bị những gì?",
        "Lịch tuyển sinh năm 2026 như thế nào?",
        "Xét tuyển bổ sung HaUI có không?",
    ],
    IntentType.RAG_TRUONG_HOC_BONG: [
        "HaUI có những loại học bổng gì?",
        "Điều kiện để nhận học bổng khuyến khích học tập",
        "Ký túc xá HaUI như thế nào, giá bao nhiêu?",
        "Trường đại học công nghiệp Hà Nội ở đâu?",
        "Cơ sở vật chất của HaUI có tốt không?",
        "Giới thiệu chung về trường HaUI",
        "HaUI có bao nhiêu sinh viên, bao nhiêu khoa?",
        # Thêm các câu người dùng hay hỏi mà v2 miss:
        "Trường thành viên HaUI gồm những trường nào?",
        "Cơ cấu tổ chức đào tạo của HaUI",
        "HaUI có bao nhiêu cơ sở đào tạo?",
        "Có mấy khoa trong trường HaUI?",
        "Các đơn vị trực thuộc HaUI là gì?",
        "HaUI gồm có những trường và khoa nào?",
        "Sơ đồ tổ chức của Đại học Công nghiệp Hà Nội",
        "GPA bao nhiêu thì được học bổng HaUI?",
        "Điều kiện xét học bổng KKHT",
        "Sinh viên nghèo có được học bổng gì?",
        "Học bổng Nguyễn Thanh Bình dành cho đối tượng nào?",
    ],
    IntentType.GREETING: [
        "Xin chào bạn", "Hello", "Chào HaUI bot",
        "Cảm ơn bạn đã giúp", "Bạn là ai vậy?",
        "Tạm biệt nhé", "Cho tôi hỏi chút", "Hi bot",
    ],
    IntentType.OFF_TOPIC: [
        "Hôm nay thời tiết thế nào?",
        "Cho tôi công thức nấu phở",
        "Giá bitcoin hôm nay bao nhiêu?",
        "Phim hay nhất năm 2024 là gì?",
        "Viết code Python cho tôi",
    ],
}


class EmbeddingClassifier:
    """Fallback khi LLM không available — giữ nguyên logic từ v2."""

    CONFIDENCE_THRESHOLD = 0.55

    def __init__(self, embed_fn):
        self._embed_fn   = embed_fn
        self._built      = False
        self._intent_vecs: dict[IntentType, np.ndarray] = {}

    def build(self) -> None:
        logger.info("EmbeddingClassifier: building intent vectors...")
        for intent_type, examples in INTENT_EXAMPLES.items():
            vecs  = np.array([self._embed_fn(ex) for ex in examples], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._intent_vecs[intent_type] = vecs / norms
        self._built = True
        logger.info(f"EmbeddingClassifier: built {len(self._intent_vecs)} intents")

    def classify(self, query: str) -> tuple[IntentType, float]:
        if not self._built:
            return IntentType.UNKNOWN, 0.0

        q_vec = np.array(self._embed_fn(query), dtype=np.float32)
        norm  = np.linalg.norm(q_vec)
        if norm == 0:
            return IntentType.UNKNOWN, 0.0
        q_vec = q_vec / norm

        best_intent = IntentType.UNKNOWN
        best_score  = 0.0

        for intent_type, vecs in self._intent_vecs.items():
            sims      = vecs @ q_vec
            max_sim   = float(sims.max())
            mean_top3 = float(np.sort(sims)[-3:].mean())
            score     = 0.7 * max_sim + 0.3 * mean_top3

            if score > best_score:
                best_score  = score
                best_intent = intent_type

        if best_score < self.CONFIDENCE_THRESHOLD:
            return IntentType.UNKNOWN, best_score

        return best_intent, round(best_score, 3)


# ── Router v3 ─────────────────────────────────────────────────────────────────

class Router:
    """
    Router v3 — LLM-first classification.

    Pipeline:
      1. Fast-path rules  → GREETING / OFF_TOPIC / rõ ràng có số liệu   (< 1ms)
      2. LLM classify     → hiểu ngữ nghĩa, không cần pattern           (~150ms)
      3. Embedding fallback → khi LLM timeout/unavailable               (~20ms)
      4. UNKNOWN fallback → cuối cùng, đẩy vào RAG

    Backward compatible 100% với chatbot.py — chỉ cần đổi import.
    """

    def __init__(self):
        self._llm_clf  : Optional[LLMClassifier]       = None
        self._embed_clf: Optional[EmbeddingClassifier] = None

    def init_llm(self, base_url: str = OLLAMA_BASE_URL, model: str = ROUTER_MODEL) -> None:
        """Khởi tạo LLM classifier. Gọi sau khi Ollama đã sẵn sàng."""
        self._llm_clf = LLMClassifier(base_url=base_url, model=model)
        logger.info(f"Router: LLM classifier initialized (model={model})")

    def init_embedder(self, embed_fn) -> None:
        """Khởi tạo embedding fallback. Gọi sau khi Retriever/Embedder load xong."""
        self._embed_clf = EmbeddingClassifier(embed_fn)
        self._embed_clf.build()
        logger.info("Router: embedding fallback initialized")

    def classify(self, query: str) -> Intent:
        entities = _extract_entities_rule(query)

        # Bước 1: Fast-path — không cần LLM cho những câu cực rõ
        fast = _fast_path(query)
        if fast:
            intent_type, confidence = fast
            logger.debug(f"Router fast-path: {intent_type} ← '{query[:40]}'")
            return Intent(
                intent_type=intent_type,
                confidence=confidence,
                entities=entities,
                method="rule",
            )

        # Bước 2: LLM classify — thông minh thật sự
        if self._llm_clf:
            result = self._llm_clf.classify(query)
            if result:
                intent_type, confidence = result
                logger.debug(f"Router LLM: {intent_type} ({confidence:.2f}) ← '{query[:40]}'")
                return Intent(
                    intent_type=intent_type,
                    confidence=confidence,
                    entities=entities,
                    method="llm",
                )

        # Bước 3: Embedding fallback
        if self._embed_clf and self._embed_clf._built:
            intent_type, confidence = self._embed_clf.classify(query)
            if confidence >= 0.55:
                logger.debug(f"Router embed: {intent_type} ({confidence:.2f}) ← '{query[:40]}'")
                return Intent(
                    intent_type=intent_type,
                    confidence=confidence,
                    entities=entities,
                    method="embed",
                )

        # Bước 4: UNKNOWN → RAG
        return Intent(
            intent_type=IntentType.UNKNOWN,
            confidence=0.0,
            entities=entities,
            method="fallback",
        )

    def classify_batch(self, queries: list[str]) -> list[Intent]:
        return [self.classify(q) for q in queries]


# ── Backward compat: _rule_match vẫn được dùng ở chatbot.py ─────────────────

def _rule_match(query: str) -> tuple[IntentType, float] | None:
    """
    Backward-compatible wrapper.
    chatbot.py gọi _rule_match cho fast-path greeting/off-topic.
    """
    return _fast_path(query)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time as _time

    router = Router()
    router.init_llm()

    test_cases = [
        # Những câu v2 hay miss
        "trường thành viên haui",
        "cơ cấu tổ chức đào tạo",
        "có bao nhiêu cơ sở đào tạo",
        "HaUI gồm có những khoa nào?",
        "GPA bao nhiêu được học bổng KKHT?",
        # Câu bình thường
        "Điểm chuẩn ngành CNTT năm 2024",
        "Học phí ngành cơ khí bao nhiêu?",
        "25 điểm KV1 có đậu CNTT không?",
        "Xin chào",
        "Hôm nay thời tiết thế nào?",
    ]

    print(f"{'Câu hỏi':<50} {'Intent':<22} {'Conf':>5} {'Method'}")
    print("─" * 95)
    for q in test_cases:
        t0     = _time.perf_counter()
        intent = router.classify(q)
        ms     = (_time.perf_counter() - t0) * 1000
        print(
            f"{q[:48]:<50} "
            f"{intent.intent_type.value:<22} "
            f"{intent.confidence:>5.2f} "
            f"{intent.method:<10} "
            f"({ms:.0f}ms)"
        )