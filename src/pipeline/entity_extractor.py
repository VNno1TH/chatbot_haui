"""
entity_extractor.py — Entity extraction v2

Giải quyết: P1 - Entity regex yếu, không bắt được:
  - "Trường Cơ khí - Ô tô có bao nhiêu ngành"
  - "các ngành trong trường kinh tế"
  - "ngành nào thuộc khoa CNTT"

Thêm khả năng nhận dạng:
  - ten_truong / ten_khoa entity
  - Phân biệt câu hỏi về trường vs câu hỏi về ngành cụ thể
"""

from __future__ import annotations
import re
from typing import Optional


# ── Khoa / Trường patterns ───────────────────────────────────────────────────
#
# Bắt các dạng:
#   "trường CNTT", "khoa Cơ khí", "Trường Kinh tế",
#   "Trường Cơ khí - Ô tô", "khoa công nghệ hóa"

_TRUONG_KHOA_TRIGGER = re.compile(
    r"(?:trường|khoa)\s+(.{3,60}?)(?:\s+(?:có|gồm|là|của|thì|thuộc|bao nhiêu|danh sách|ngành|điểm|học phí)|\?|$)",
    re.IGNORECASE | re.UNICODE,
)

_TRUONG_WITHIN_PATTERN = re.compile(
    r"(?:trong|thuộc|của)\s+(?:trường|khoa)\s+(.{3,60}?)(?:\s+(?:có|gồm|là|điểm|học phí)|\?|$)",
    re.IGNORECASE | re.UNICODE,
)

_NGANH_THUOC_PATTERN = re.compile(
    r"ngành\s+(?:nào|của|thuộc)\s+(?:trường|khoa)\s+(.{3,60}?)(?:\s+\?|$)",
    re.IGNORECASE | re.UNICODE,
)

# Patterns cho câu hỏi "trường ... có bao nhiêu ngành"
_BAO_NHIEU_NGANH = re.compile(
    r"(?:trường|khoa)\s+(.{3,60}?)\s+(?:có|gồm)\s+(?:bao nhiêu|mấy)\s+ngành",
    re.IGNORECASE | re.UNICODE,
)


def extract_truong_khoa(query: str) -> Optional[str]:
    """
    Trích xuất tên trường/khoa từ câu hỏi.
    
    Trả về tên trường/khoa nếu tìm thấy, None nếu không.
    
    Ví dụ:
      "Trường Cơ khí - Ô tô có bao nhiêu ngành?" → "Cơ khí - Ô tô"
      "các ngành trong trường kinh tế"            → "kinh tế"
      "khoa CNTT có những ngành gì"               → "CNTT"
    """
    for pattern in [_BAO_NHIEU_NGANH, _TRUONG_KHOA_TRIGGER,
                    _TRUONG_WITHIN_PATTERN, _NGANH_THUOC_PATTERN]:
        m = pattern.search(query)
        if m:
            raw = m.group(1).strip()
            # Cắt bỏ trailing stop words
            for stop in ["có", "gồm", "là", "thì", "thuộc", "bao nhiêu",
                         "mấy", "nào", "và", "với"]:
                if raw.lower().endswith(" " + stop):
                    raw = raw[:-(len(stop)+1)].strip()
            if len(raw) >= 2:
                return raw
    return None


def extract_entities_v2(query: str) -> dict:
    """
    Entity extraction v2 — mạnh hơn rule-based cũ.
    
    Thêm:
      - ten_truong/ten_khoa: tên trường hoặc khoa
      - is_truong_query: True nếu câu hỏi về trường/khoa
    
    Backward compatible với v1 (giữ nguyên các field cũ).
    """
    q        = query.lower().strip()
    entities = {}

    # ── Các field từ v1 (giữ nguyên) ─────────────────────────────────────────

    # Năm
    nam_match = re.search(r"năm (202[3-9]|20[3-9]\d)", q)
    if nam_match:
        entities["nam"] = int(nam_match.group(1))

    # Phương thức
    pt_match = re.search(r"pt([1-5])", q)
    if pt_match:
        entities["phuong_thuc"] = f"PT{pt_match.group(1)}"

    # Tổ hợp môn
    to_hop_match = re.search(r"\b([abcdx]\d{2}|dd2)\b", q)
    if to_hop_match:
        entities["to_hop"] = to_hop_match.group(1).upper()

    # Khu vực
    kv_patterns = [
        (r"kv2[-\s]nt\b|khu\s*v[ựu]c\s*2[-\s]nt\b", "KV2-NT"),
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
        r"(?:được|thi|đạt|là|scored?)?\s*"
        r"(\d{1,3}[.,]\d{1,2}|\d{3}(?![\d.,])|\d{2}(?![\d.,]))"
        r"\s*(?:điểm|đ\b)?",
        q
    )
    if diem_match:
        entities["diem"] = float(diem_match.group(1).replace(",", "."))

    # ── Ngành (v1 map giữ nguyên) ─────────────────────────────────────────────
    NGANH_MAP = [
        (r"kỹ thuật phần mềm|\bktpm\b",                       "Kỹ thuật phần mềm"),
        (r"khoa học máy tính|\bksmt\b",                        "Khoa học máy tính"),
        (r"công nghệ thông tin|\bcntt\b",                      "Công nghệ thông tin"),
        (r"robot\b|trí tuệ nhân tạo|tự động hóa thông minh",  "Robot và trí tuệ nhân tạo"),
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
        (r"hóa học|công nghệ hóa|kỹ thuật hóa",               "Công nghệ kỹ thuật hóa học"),
        (r"hóa dược|dược",                                     "Hóa dược"),
    ]
    for pattern, ten_chuan in NGANH_MAP:
        if re.search(pattern, q):
            entities["nganh"] = ten_chuan
            break

    # ── [MỚI] Trường / Khoa detection ─────────────────────────────────────────
    ten_truong = extract_truong_khoa(query)
    if ten_truong:
        entities["ten_truong"] = ten_truong
        entities["is_truong_query"] = True

    # Detect câu hỏi về cơ cấu tổ chức tổng quan
    CO_CAU_PATTERNS = [
        r"bao nhiêu\s+(?:trường|khoa)",
        r"(?:trường|khoa)\s+nào",
        r"(?:có|gồm)\s+(?:những|các)\s+(?:trường|khoa)",
        r"cơ cấu\s+(?:tổ chức|đào tạo)",
        r"các\s+đơn vị\s+trực thuộc",
        r"thành viên\s+(?:của\s+)?haui",
        r"trường\s+trực thuộc",
        r"đơn vị\s+(?:trực thuộc|thành viên)",
    ]
    if any(re.search(p, q) for p in CO_CAU_PATTERNS):
        entities["is_co_cau_query"] = True

    return entities


def is_truong_khoa_query(query: str, entities: dict) -> bool:
    """
    True nếu câu hỏi hỏi về trường/khoa (ngành thuộc trường, ...)
    thay vì hỏi về ngành cụ thể.
    """
    return (
        entities.get("is_truong_query", False)
        or entities.get("is_co_cau_query", False)
        or bool(entities.get("ten_truong"))
    )