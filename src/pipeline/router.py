"""
router.py  (v2 — LOCAL, không dùng Claude)
Phân loại câu hỏi của user → quyết định dùng query_json hay ChromaDB.

Chiến lược:
  1. Rule-based (keyword/regex)  — < 1ms, bắt ~80% câu hỏi thông thường
  2. Embedding similarity         — ~20ms, bắt câu tự nhiên không có keyword
     Dùng lại model bge-m3 (hoặc multilingual-e5-small nếu muốn nhẹ hơn).
     Mỗi intent có ~8-12 câu ví dụ được embed sẵn lúc khởi động.
  3. Fallback UNKNOWN             → đẩy thẳng vào RAG, không gọi LLM nào.

Không còn phụ thuộc anthropic / Claude.
"""

from __future__ import annotations

import json
import re
import logging
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("haui.router")

# ── Định nghĩa intent ─────────────────────────────────────────────────────────

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
    method      : str  = "rule"   # "rule" | "embed" | "fallback"

    @property
    def is_json(self) -> bool:
        return self.intent_type in JSON_INTENTS

    @property
    def is_rag(self) -> bool:
        return self.intent_type in RAG_INTENTS


# ── Rule-based patterns ───────────────────────────────────────────────────────

RULES: list[tuple[IntentType, list[str]]] = [

    (IntentType.JSON_DAU_TRUOT, [
        r"có đậu", r"có vào được", r"đỗ không", r"trượt không",
        r"đủ điểm", r"thiếu mấy điểm", r"có đủ", r"có trúng tuyển",
        r"có trúng", r"đủ để vào", r"vào được không", r"qua không",
        r"\d+\s*điểm.*vào", r"vào.*\d+\s*điểm",
        r"tư vấn.*điểm", r"với điểm này", r"điểm này.*được không",
    ]),

    (IntentType.JSON_QUY_DOI_DIEM, [
        r"quy đổi", r"điểm hsa", r"điểm tsa", r"đánh giá năng lực",
        r"đánh giá tư duy", r"học bạ.*quy", r"ưu tiên khu vực",
        r"điểm ưu tiên", r"kv1", r"kv2", r"kv3", r"đối tượng 0[1-6]",
        r"tính điểm", r"cộng điểm",
        r"được bao nhiêu điểm", r"tính ra bao nhiêu", r"điểm xét tuyển là bao nhiêu",
        r"khu vực.*điểm", r"điểm.*khu vực", r"ưu tiên.*được",
        r"tư duy.*khu vực", r"khu vực.*tư duy",
        r"cộng mấy", r"được cộng",
    ]),

    (IntentType.JSON_DIEM_CHUAN, [
        r"điểm chuẩn", r"điểm đầu vào",
        r"cần mấy điểm", r"điểm tối thiểu",
        r"năm ngoái.*điểm", r"điểm.*năm 202[3-6]",
        r"điểm chuẩn.*bao nhiêu", r"bao nhiêu.*điểm chuẩn",
        r"so sánh.*điểm", r"xu hướng.*điểm", r"lịch sử.*điểm",
        r"3 năm", r"qua các năm", r"nhiều năm", r"các năm",
        r"tăng.*điểm", r"giảm.*điểm", r"biến động",
    ]),

    (IntentType.JSON_HOC_PHI, [
        r"học phí.*bao nhiêu", r"bao nhiêu.*học phí",
        r"tiền học", r"chi phí học", r"đóng tiền",
        r"một tín chỉ", r"mỗi tín chỉ", r"học kỳ.*tiền",
        r"học phí.*tín chỉ", r"tín chỉ.*tiền",
        r"chi phí.*4 năm", r"chi phí.*toàn khóa",
    ]),

    (IntentType.JSON_CHI_TIEU_TO_HOP, [
        r"tổ hợp", r"môn thi", r"xét tuyển bằng", r"chỉ tiêu",
        r"bao nhiêu chỉ tiêu", r"tuyển bao nhiêu", r"pt[1-5]",
        r"phương thức [1-5]", r"phương thức xét tuyển", r"phương thức nào",
        r"d0[1-9]", r"a0[0-9]", r"x0[0-9]", r"x2[5-7]", r"dd2",
    ]),

    (IntentType.RAG_TRUONG_HOC_BONG, [
        r"học bổng", r"giới thiệu trường", r"lịch sử trường",
        r"về trường", r"haui là", r"đại học công nghiệp",
        r"cơ sở vật chất", r"khuôn viên", r"ký túc xá",
        r"chính sách ưu đãi", r"chính sách hỗ trợ",
        r"haui ở đâu", r"trường có bao nhiêu",
    ]),

    (IntentType.RAG_FAQ, [
        r"đăng ký", r"nộp hồ sơ", r"nhập học", r"thủ tục",
        r"hướng dẫn", r"cách đăng", r"làm thế nào", r"khi nào",
        r"thời gian", r"lịch", r"deadline", r"xét tuyển.*khi",
        r"hồ sơ gồm", r"cần chuẩn bị",
        r"lịch tuyển sinh", r"ngày.*nộp", r"hạn.*nộp",
    ]),

    (IntentType.RAG_MO_TA_NGANH, [
        r"học gì", r"học những gì", r"ra làm gì", r"việc làm",
        r"cơ hội", r"triển vọng", r"ngành.*gồm", r"chương trình học",
        r"môn học", r"chuẩn đầu ra", r"sau khi tốt nghiệp",
        r"ngành.*có gì", r"giới thiệu ngành",
        r"so sánh.*ngành", r"ngành.*so sánh", r"hay.*ngành",
        r"ngành nào.*tốt", r"nên chọn ngành", r"khác nhau.*ngành",
        r"có gì hay", r"đặc biệt.*ngành", r"ngành.*đặc biệt",
        r"tương lai", r"thu nhập", r"lương",
    ]),

    (IntentType.GREETING, [
        r"^(xin )?chào", r"^hello", r"^hi\b", r"^hey\b",
        r"^ơi\b", r"^alo\b",
        r"^cảm ơn", r"^thanks", r"^thank",
        r"^ok\b", r"^okay\b", r"^được rồi",
        r"^bye\b", r"^tạm biệt", r"^hẹn gặp",
        r"bạn là ai", r"bạn tên (là )?gì", r"bạn là (gì|ai)",
        r"bạn làm được gì", r"bạn có thể (giúp|làm)",
        r"cho tôi hỏi", r"^hỏi.*chút",
        r"^good (morning|afternoon|evening|night)",
        r"bạn có khỏe", r"bạn thế nào",
    ]),

    (IntentType.OFF_TOPIC, [
        r"thời tiết", r"dự báo", r"mưa.*nay", r"nắng.*nay",
        r"tin tức", r"tin mới", r"bóng đá", r"thể thao",
        r"nấu.*gì", r"món ăn", r"công thức nấu", r"nhà hàng", r"quán ăn",
        r"phim.*hay", r"bài hát", r"ca sĩ", r"diễn viên",
        r"chứng khoán", r"bitcoin", r"crypto", r"giá vàng", r"tỷ giá",
        r"bệnh viện", r"thuốc.*gì", r"triệu chứng", r"chữa bệnh",
        r"viết code", r"debug", r"lỗi code", r"javascript", r"python.*lỗi",
        r"fix.*bug", r"câu lệnh",
        r"bạn trai", r"bạn gái", r"tình yêu", r"chia tay",
        r"tôi buồn", r"tôi vui", r"tôi mệt",
    ]),
]


def _normalize(text: str) -> str:
    return text.lower().strip()


def _rule_match(query: str) -> tuple[IntentType, float] | None:
    q = _normalize(query)
    scores: dict[IntentType, int] = {}

    for intent_type, patterns in RULES:
        count = sum(1 for p in patterns if re.search(p, q))
        if count > 0:
            scores[intent_type] = count

    if not scores:
        return None

    best_intent = max(scores, key=lambda k: scores[k])
    total_match = sum(scores.values())
    best_score  = scores[best_intent]
    confidence  = min(0.95, 0.5 + (best_score / max(total_match, 1)) * 0.5)

    top_scores = [v for v in scores.values() if v == best_score]
    if len(top_scores) > 1:
        confidence *= 0.7

    return best_intent, confidence


# ── Entity extraction ─────────────────────────────────────────────────────────

def _extract_entities_rule(query: str) -> dict:
    q        = _normalize(query)
    entities = {}

    # Năm
    nam_match = re.search(r"năm (202[3-9]|20[3-9]\d)", q)
    if nam_match:
        entities["nam"] = int(nam_match.group(1))

    # Phương thức
    pt_match = re.search(r"pt([1-5])", q)
    if pt_match:
        entities["phuong_thuc"] = f"PT{pt_match.group(1)}"

    # Tổ hợp
    to_hop_match = re.search(r"\b([abcdx]\d{2}|dd2)\b", q)
    if to_hop_match:
        entities["to_hop"] = to_hop_match.group(1).upper()

    # Khu vực
    kv_patterns = [
        (r"kv2[-\s]nt\b|kv2-nt\b|khu\s*v[ựu]c\s*2[-\s]nt\b|khu\s*v[ựu]c\s*2\s*n[oô]ng\s*th[oô]n", "KV2-NT"),
        (r"\bkv1\b|khu\s*v[ựu]c\s*1\b", "KV1"),
        (r"\bkv2\b|khu\s*v[ựu]c\s*2\b", "KV2"),
        (r"\bkv3\b|khu\s*v[ựu]c\s*3\b", "KV3"),
    ]
    for _pattern, _kv_value in kv_patterns:
        if re.search(_pattern, q):
            entities["khu_vuc"] = _kv_value
            break

    # Đối tượng ưu tiên
    dt_match = re.search(r"đối tượng 0([1-6])", q)
    if dt_match:
        entities["doi_tuong"] = f"0{dt_match.group(1)}"

    # Điểm số
    diem_match = re.search(
        r"(?:được|thi|đạt|đạt được|là|scored?)?\s*"
        r"(\d{1,3}[.,]\d{1,2}|\d{3}(?![\d.,])|\d{2}(?![\d.,]))"
        r"\s*(?:điểm|đ\b)?",
        q
    )
    if diem_match:
        entities["diem"] = float(diem_match.group(1).replace(",", "."))

    # Tên ngành
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


# ── Embedding-based fallback ──────────────────────────────────────────────────
# Câu ví dụ cho mỗi intent — đa dạng cách diễn đạt tự nhiên
# Được embed 1 lần lúc khởi động, dùng mãi không cần gọi LLM

INTENT_EXAMPLES: dict[IntentType, list[str]] = {
    IntentType.JSON_DIEM_CHUAN: [
        "Điểm chuẩn ngành CNTT năm 2024 là bao nhiêu?",
        "Năm ngoái vào ngành cơ khí cần bao nhiêu điểm?",
        "Điểm đầu vào ngành kế toán HaUI",
        "Điểm chuẩn các năm của ngành điện tử",
        "Xu hướng điểm chuẩn ngành logistics tăng hay giảm?",
        "Năm 2023 ngành KTPM cần bao nhiêu để đỗ?",
        "Cho tôi xem lịch sử điểm chuẩn ngành quản trị kinh doanh",
        "Điểm vào ngành robot trí tuệ nhân tạo mấy điểm",
        "Ngành marketing lấy bao nhiêu điểm",
        "Điểm chuẩn 2025 ngành cơ điện tử",
        "Cần mấy điểm để vào HaUI ngành CNTT",
        "Điểm chuẩn PT3 ngành kế toán 2024",
        "So sánh điểm chuẩn CNTT 3 năm gần đây",
        "Điểm thi THPT vào ngành điện điện tử HaUI",
    ],
    IntentType.JSON_HOC_PHI: [
        "Học phí ngành CNTT bao nhiêu một tín chỉ?",
        "Chi phí học 4 năm tại HaUI khoảng bao nhiêu?",
        "Tiền học mỗi học kỳ ngành kỹ thuật điện là bao nhiêu?",
        "Đóng học phí mỗi năm hết bao nhiêu?",
        "Ngành cơ khí học phí có đắt không?",
        "Một tín chỉ ngành kinh tế tốn bao nhiêu tiền?",
        "Học phí chương trình chất lượng cao khác gì đại trà?",
        "Tổng chi phí toàn khóa ngành kế toán là bao nhiêu?",
        "1 tín chỉ HaUI bao nhiêu tiền",
        "Học phí K20 đại trà là bao nhiêu",
        "K19 đóng tiền học bao nhiêu một tín",
        "Tiền học K18 một tín chỉ",
        "Học phí năm học 2025 2026 HaUI",
        "Học bằng tiếng Anh tốn bao nhiêu tiền",
    ],
    IntentType.JSON_CHI_TIEU_TO_HOP: [
        "Ngành CNTT xét tuyển tổ hợp môn nào?",
        "Học toán lý hóa có xét tuyển được ngành điện tử không?",
        "Chỉ tiêu tuyển sinh ngành logistics năm nay là bao nhiêu?",
        "Phương thức xét tuyển của HaUI gồm những gì?",
        "Tổ hợp A00 có xét tuyển được ngành nào?",
        "Ngành cơ khí tuyển bao nhiêu chỉ tiêu?",
        "Xét tuyển học bạ thì cần tổ hợp gì?",
        "Năm 2026 HaUI tuyển tổng bao nhiêu sinh viên?",
        "Ngành kế toán dùng tổ hợp D01 được không",
        "Tổ hợp X06 X07 gồm những môn gì",
        "HaUI có mấy phương thức xét tuyển",
        "Chỉ tiêu ngành robot 2025",
        "Tôi học khối A có vào được CNTT không",
        "Ngành tiếng Anh xét tổ hợp gì",
    ],
    IntentType.JSON_QUY_DOI_DIEM: [
        # Quy đổi HSA
        "Điểm TSA 850 quy đổi được bao nhiêu điểm?",
        "HSA 105 tương đương mấy điểm thang 30?",
        "Điểm học bạ 8.5 quy đổi thế nào?",
        "Đánh giá tư duy 700 điểm được bao nhiêu?",
        "Điểm xét tuyển của tôi sẽ là bao nhiêu nếu học bạ 8.0?",
        "90 điểm hsa thì được bao nhiêu điểm",
        "80 tsa khu vực 1 thì điểm xét tuyển là bao nhiêu",
        "tôi được 95 hsa khu vực 2 nông thôn",
        "hsa 110 kv1 điểm xét tuyển bao nhiêu",
        "tsa 75 được mấy điểm thang 30",
        # Ưu tiên khu vực
        "Tôi ở khu vực 1 được cộng mấy điểm ưu tiên?",
        "KV2-NT được cộng thêm bao nhiêu điểm?",
        "Tính điểm ưu tiên đối tượng 01 như thế nào?",
        "25 điểm KV1 thì điểm ưu tiên thực tế là bao nhiêu",
        "khu vực 2 nông thôn cộng mấy điểm",
        "22 điểm khu vực 1 được cộng bao nhiêu",
        "điểm ưu tiên KV2 là bao nhiêu",
        "tôi thuộc đối tượng 03 được cộng mấy điểm",
        "30 điểm có được cộng ưu tiên không",
    ],
    IntentType.JSON_DAU_TRUOT: [
        "Điểm 24.5 KV1 có đậu ngành CNTT không?",
        "Mình thi được 26 điểm, vào ngành điện tử được không?",
        "TSA 800 có đủ để vào ngành kỹ thuật phần mềm không?",
        "Em 23 điểm KV2-NT, trúng tuyển ngành kế toán được không?",
        "Học bạ 8.2 có vào được HaUI ngành cơ khí không?",
        "Tôi đạt 25 điểm THPT, có trượt ngành CNTT không?",
        "Điểm này có đủ vào ngành logistics không?",
        "Với điểm HSA 110, em có khả năng đậu ngành robot không?",
        "Em 23 điểm thi THPT khu vực 2 có đỗ không",
        "hsa 95 kv1 có vào được cntt không",
        "điểm học bạ 8.5 có đậu ngành marketing không",
        "25.5 điểm có trúng tuyển kỹ thuật phần mềm không",
        "em đạt 90 tsa vào ngành cơ điện tử được không",
        "với 24 điểm em có nên đăng ký ngành kế toán không",
    ],
    IntentType.RAG_MO_TA_NGANH: [
        "Ngành robot và trí tuệ nhân tạo sau khi ra trường làm gì?",
        "Học CNTT ở HaUI thì học những môn gì?",
        "Cơ hội việc làm ngành kỹ thuật điện như thế nào?",
        "Ngành logistics có triển vọng không?",
        "So sánh ngành CNTT với kỹ thuật phần mềm khác nhau thế nào?",
        "Chương trình đào tạo ngành cơ khí gồm những gì?",
        "Lương ngành kế toán sau khi ra trường bao nhiêu?",
        "Nên học ngành nào có nhiều việc làm nhất?",
        "Chuẩn đầu ra ngành an toàn thông tin là gì?",
        "Học kinh tế số ra làm gì?",
        "Ngành cơ điện tử HaUI học gì",
        "An toàn thông tin và CNTT khác nhau như thế nào",
        "Ngành ô tô HaUI có tốt không",
        "Thu nhập ngành kỹ thuật phần mềm bao nhiêu",
        "Ngành robot AI có dễ xin việc không",
    ],
    IntentType.RAG_FAQ: [
        "Làm thế nào để đăng ký xét tuyển vào HaUI?",
        "Thủ tục nhập học gồm những gì?",
        "Hạn nộp hồ sơ xét tuyển năm 2025 là khi nào?",
        "Hướng dẫn đăng ký trên cổng ĐKXT quốc gia",
        "Hồ sơ nhập học cần chuẩn bị những gì?",
        "Lịch tuyển sinh năm 2026 như thế nào?",
        "Xét tuyển bổ sung HaUI có không?",
        "Deadline nộp hồ sơ nguyện vọng là ngày mấy?",
        "Thí sinh tự do có được đăng ký PT4 PT5 không",
        "Nộp hồ sơ online hay trực tiếp",
        "Lệ phí đăng ký xét tuyển HaUI bao nhiêu",
        "Trúng tuyển rồi nhập học như thế nào",
        "Có thể sửa nguyện vọng sau khi đăng ký không",
        "Thời gian nhập học năm 2026",
    ],
    IntentType.RAG_TRUONG_HOC_BONG: [
        "HaUI có những loại học bổng gì?",
        "Điều kiện để nhận học bổng khuyến khích học tập",
        "Ký túc xá HaUI như thế nào, giá bao nhiêu?",
        "Trường đại học công nghiệp Hà Nội ở đâu?",
        "Cơ sở vật chất của HaUI có tốt không?",
        "Giới thiệu chung về trường HaUI",
        "Chính sách hỗ trợ sinh viên của HaUI",
        "HaUI có bao nhiêu sinh viên, bao nhiêu khoa?",
        "Học bổng toàn khóa HaUI cần điều kiện gì",
        "Phòng ký túc xá HaUI giá bao nhiêu một tháng",
        "HaUI trực thuộc bộ nào",
        "Mã trường HaUI là gì",
        "HaUI có mấy cơ sở",
        "Tỷ lệ sinh viên HaUI có việc làm sau tốt nghiệp",
    ],
    IntentType.GREETING: [
        "Xin chào bạn",
        "Hello",
        "Chào HaUI bot",
        "Cảm ơn bạn đã giúp",
        "Bạn là ai vậy?",
        "Bạn có thể giúp tôi không?",
        "Tạm biệt nhé",
        "Cho tôi hỏi chút",
        "Hi bot",
        "Ơi cho hỏi",
        "Chào buổi sáng",
        "thanks nha",
    ],
    IntentType.OFF_TOPIC: [
        "Hôm nay thời tiết thế nào?",
        "Cho tôi công thức nấu phở",
        "Giá bitcoin hôm nay bao nhiêu?",
        "Tôi bị đau đầu phải làm gì?",
        "Phim hay nhất năm 2024 là gì?",
        "Bóng đá tối nay kết quả ra sao?",
        "Viết code Python cho tôi",
        "Tôi đang buồn quá",
        "Recommend nhà hàng ngon ở Hà Nội",
        "ChatGPT có thể làm gì",
    ],
}


class EmbeddingClassifier:
    """
    Phân loại intent bằng cosine similarity với các câu ví dụ đã embed sẵn.
    Dùng lại model embedding của Retriever — không load thêm model nào.

    Cách dùng:
        clf = EmbeddingClassifier(embed_fn)   # embed_fn: str -> np.ndarray
        clf.build()                            # embed tất cả câu ví dụ (chạy 1 lần)
        intent_type, conf = clf.classify("Ngành CNTT học gì?")
    """

    # Ngưỡng confidence tối thiểu để dùng kết quả embedding
    CONFIDENCE_THRESHOLD = 0.55

    def __init__(self, embed_fn):
        """
        Args:
            embed_fn: callable(str) -> np.ndarray (1-D, normalized)
                      Chính là embedder.embed_query() của Embedder hiện có.
        """
        self._embed_fn   = embed_fn
        self._built      = False
        # {IntentType: np.ndarray shape (n_examples, dim)}
        self._intent_vecs: dict[IntentType, np.ndarray] = {}

    def build(self) -> None:
        """Embed tất cả câu ví dụ — gọi 1 lần lúc khởi động."""
        logger.info("EmbeddingClassifier: đang build intent vectors...")
        for intent_type, examples in INTENT_EXAMPLES.items():
            vecs = np.array([self._embed_fn(ex) for ex in examples], dtype=np.float32)
            # normalize để cosine = dot product
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._intent_vecs[intent_type] = vecs / norms
        self._built = True
        logger.info(f"EmbeddingClassifier: built {len(self._intent_vecs)} intents "
                    f"({sum(v.shape[0] for v in self._intent_vecs.values())} total examples)")

    def classify(self, query: str) -> tuple[IntentType, float]:
        """
        Trả về (IntentType, confidence).
        Nếu chưa build hoặc confidence thấp → trả về (UNKNOWN, 0.0).
        """
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
            # Cosine similarity với toàn bộ examples, lấy max
            sims      = vecs @ q_vec          # shape (n,)
            max_sim   = float(sims.max())
            mean_top3 = float(np.sort(sims)[-3:].mean())
            # Kết hợp max + mean top-3 để ổn định hơn
            score = 0.7 * max_sim + 0.3 * mean_top3

            if score > best_score:
                best_score  = score
                best_intent = intent_type

        if best_score < self.CONFIDENCE_THRESHOLD:
            return IntentType.UNKNOWN, best_score

        return best_intent, round(best_score, 3)


# ── Multi-intent detection ────────────────────────────────────────────────────

MULTI_INTENT_PATTERNS: list[tuple[list[str], list[IntentType]]] = [
    (
        [r"có đậu|đỗ không|trượt không|có vào",
         r"học phí|tiền học|tín chỉ"],
        [IntentType.JSON_DAU_TRUOT, IntentType.JSON_HOC_PHI],
    ),
    (
        [r"điểm chuẩn|điểm đầu vào",
         r"tổ hợp|môn thi|xét tuyển bằng"],
        [IntentType.JSON_DIEM_CHUAN, IntentType.JSON_CHI_TIEU_TO_HOP],
    ),
    (
        [r"điểm chuẩn|điểm đầu vào",
         r"học gì|ra làm gì|ngành.*gồm|cơ hội"],
        [IntentType.JSON_DIEM_CHUAN, IntentType.RAG_MO_TA_NGANH],
    ),
    (
        [r"học phí|tiền học",
         r"học bổng|hỗ trợ|ưu đãi"],
        [IntentType.JSON_HOC_PHI, IntentType.RAG_TRUONG_HOC_BONG],
    ),
    (
        [r"có đậu|đỗ không|trượt không",
         r"học gì|ra làm gì|ngành.*như thế nào"],
        [IntentType.JSON_DAU_TRUOT, IntentType.RAG_MO_TA_NGANH],
    ),
    (
        [r"so sánh|hay|nên chọn|khác nhau|giống nhau|tốt hơn"],
        [IntentType.RAG_MO_TA_NGANH, IntentType.JSON_DIEM_CHUAN],
    ),
    (
        [r"tư vấn|nên học|nên chọn|phù hợp",
         r"\d{2}[,.]?\d*\s*điểm|\d{2,3}\s*điểm"],
        [IntentType.JSON_DAU_TRUOT, IntentType.RAG_MO_TA_NGANH, IntentType.JSON_HOC_PHI],
    ),
]


def detect_intents(query: str) -> list[IntentType]:
    q = _normalize(query)
    intents: list[IntentType] = []

    for pattern_list, intent_list in MULTI_INTENT_PATTERNS:
        if all(re.search(p, q) for p in pattern_list):
            for it in intent_list:
                if it not in intents:
                    intents.append(it)

    if not intents:
        rule_result = _rule_match(query)
        if rule_result:
            intents.append(rule_result[0])

    return intents if intents else [IntentType.UNKNOWN]


# ── Router chính ──────────────────────────────────────────────────────────────

class Router:
    """
    Phân loại câu hỏi — hoàn toàn local, không gọi API nào.

    Pipeline:
      1. Rule-based  → confidence >= 0.65 → dùng luôn  (<1ms)
      2. Embedding similarity  → confidence >= 0.55 → dùng  (~20ms)
      3. Fallback UNKNOWN  → đẩy vào RAG

    Khởi tạo:
        router = Router()            # lazy — chưa load embedding
        router.init_embedder(fn)     # gọi sau khi Retriever đã sẵn sàng
    """

    RULE_THRESHOLD  = 0.65
    EMBED_THRESHOLD = 0.55

    def __init__(self):
        self._embed_clf: Optional[EmbeddingClassifier] = None

    def init_embedder(self, embed_fn) -> None:
        """
        Truyền hàm embed vào để build embedding classifier.
        Gọi 1 lần sau khi Retriever/Embedder đã load xong.

        Args:
            embed_fn: callable(str) -> list[float] hoặc np.ndarray
        """
        self._embed_clf = EmbeddingClassifier(embed_fn)
        self._embed_clf.build()
        logger.info("Router: embedding classifier sẵn sàng")

    def classify(self, query: str) -> Intent:
        # Bước 1: Rule-based
        rule_result = _rule_match(query)
        if rule_result is not None:
            intent_type, confidence = rule_result
            if confidence >= self.RULE_THRESHOLD:
                entities = _extract_entities_rule(query)
                return Intent(
                    intent_type = intent_type,
                    confidence  = confidence,
                    entities    = entities,
                    method      = "rule",
                )

        # Bước 2: Embedding similarity (nếu đã init)
        if self._embed_clf is not None and self._embed_clf._built:
            intent_type, confidence = self._embed_clf.classify(query)
            if confidence >= self.EMBED_THRESHOLD:
                entities = _extract_entities_rule(query)
                logger.debug(f"Router embed: {intent_type} ({confidence:.3f}) ← '{query[:50]}'")
                return Intent(
                    intent_type = intent_type,
                    confidence  = confidence,
                    entities    = entities,
                    method      = "embed",
                )

        # Bước 3: Fallback — đẩy thẳng vào RAG
        entities = _extract_entities_rule(query)
        return Intent(
            intent_type = IntentType.UNKNOWN,
            confidence  = 0.0,
            entities    = entities,
            method      = "fallback",
        )

    def classify_batch(self, queries: list[str]) -> list[Intent]:
        return [self.classify(q) for q in queries]


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    # Test không có embedder (chỉ rule-based)
    router = Router()

    test_cases = [
        "Điểm chuẩn ngành Công nghệ thông tin năm 2024 là bao nhiêu?",
        "Học phí ngành kỹ sư K3 bao nhiêu một tín chỉ?",
        "Ngành kế toán xét tuyển tổ hợp nào?",
        "Điểm HSA 105 quy đổi được bao nhiêu điểm?",
        "Mình 24.5 điểm KV1 có đậu ngành CNTT không?",
        "Ngành Robot và trí tuệ nhân tạo ra làm gì sau khi tốt nghiệp?",
        "Hướng dẫn đăng ký xét tuyển năm 2025 như thế nào?",
        "HaUI có những loại học bổng gì?",
        # Câu tự nhiên không keyword rõ — sẽ fallback nếu không có embedder
        "Năm ngoái vào ngành cơ điện tử cần bao nhiêu?",
        "Mình học xong THPT ở nông thôn, muốn vào ngành CNTT thì cần chuẩn bị gì?",
        "Chi phí học 4 năm khoảng bao nhiêu?",
    ]

    print(f"{'Câu hỏi':<55} {'Intent':<22} {'Conf':>5} {'Method'}")
    print("─" * 100)
    for q in test_cases:
        t0     = time.perf_counter()
        intent = router.classify(q)
        ms     = (time.perf_counter() - t0) * 1000
        print(
            f"{q[:52]:<55} "
            f"{intent.intent_type.value:<22} "
            f"{intent.confidence:>5.2f} "
            f"{intent.method:<10} "
            f"({ms:.0f}ms)"
        )
        if intent.entities:
            print(f"  └─ entities: {intent.entities}")