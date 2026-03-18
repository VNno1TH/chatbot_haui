"""
_utils.py  (private)
Tiện ích dùng chung trong các module query_json.
"""


def normalize(text: str) -> str:
    """Lowercase + strip để so sánh không phân biệt hoa thường."""
    return text.lower().strip()


def match_nganh(query: str, ten_nganh: str) -> bool:
    """
    Kiểm tra query có khớp tên ngành không.
    Hỗ trợ: tìm kiếm một chiều + word-level matching cho tên ngắn.
    """
    q = normalize(query)
    d = normalize(ten_nganh)
    if q in d or d in q:
        return True
    # Token matching: tách từ để bắt "logistics" khớp với "logistics và quản lý chuỗi cung ứng"
    q_words = set(q.split())
    d_words = set(d.split())
    # Nếu query ngắn (1-2 từ) → match nếu tất cả từ trong query có trong tên ngành
    if len(q_words) <= 2 and q_words and q_words.issubset(d_words):
        return True
    return False


def not_found(msg: str) -> dict:
    """Trả về dict chuẩn khi không tìm thấy kết quả."""
    return {"found": False, "thong_bao": msg}


def ok(**kwargs) -> dict:
    """Trả về dict chuẩn khi tìm thấy kết quả."""
    return {"found": True, **kwargs}