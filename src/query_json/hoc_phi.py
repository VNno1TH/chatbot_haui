"""
hoc_phi.py
Tra cứu học phí theo chương trình đào tạo.

Nguồn dữ liệu: muc_thu_hoc_phi.json
"""

import re
from ._loader import load
from ._utils  import normalize, not_found, ok


def get_hoc_phi(query: str = "") -> dict:
    """
    Lấy thông tin học phí.

    Args:
        query: Từ khóa tìm kiếm.
               VD: "cử nhân", "kỹ sư", "tiếng anh", "k20", "thạc sĩ".
               Để trống = trả về toàn bộ.

    Returns:
        {
            "found"   : True,
            "nam_hoc" : str,
            "ket_qua" : [{
                "nhom", "chuong_trinh",
                "gia_tri", "don_vi", "hien_thi"
            }]
        }
    """
    raw = load("hoc_phi")
    ds  = raw["hoc_phi_theo_chuong_trinh"]
    nam = raw.get("nam_hoc", "2025-2026")

    if query:
        q  = normalize(query)
        # Tách query thành các từ khóa quan trọng để match linh hoạt
        # Tránh lỗi: câu hỏi dài hơn tên chương trình → q IN ct = False
        _KW_PATTERN = r'k\d+|tiếng anh|đại trà|thạc sĩ|tiến sĩ|cao đẳng|liên thông|từ xa|kỹ sư|cử nhân'
        def _match(d: dict) -> int:
            """Trả về số token match — 0 = không match."""
            ct   = normalize(d["chuong_trinh"])
            nhom = normalize(d.get("nhom", ""))
            # Exact: q nằm trong tên hoặc tên nằm trong q
            if q in ct or ct in q:
                return 99
            # Token match: tách tên chương trình → tìm trong query
            tokens = re.findall(_KW_PATTERN, ct)
            return sum(1 for tok in tokens if tok in q)
        ds = [d for d in ds if _match(d) > 0]
        # Sắp xếp: match nhiều token hơn lên trước
        ds = sorted(ds, key=_match, reverse=True)

    if not ds:
        return not_found(f"Không tìm thấy thông tin học phí cho '{query}'.")

    return ok(
        nam_hoc  = nam,
        ket_qua  = [
            {
                "nhom"        : d.get("nhom", ""),
                "chuong_trinh": d["chuong_trinh"],
                "gia_tri"     : d["gia_tri"],
                "don_vi"      : d["don_vi"],
                "hien_thi"    : f"{d['gia_tri']:,} {d['don_vi']}".replace(",", "."),
            }
            for d in ds
        ],
    )


def get_hoc_phi_dai_hoc() -> dict:
    """Lấy học phí hệ đại học chính quy (K18 - K20)."""
    return get_hoc_phi("cử nhân")


def get_hoc_phi_sau_dai_hoc() -> dict:
    """Lấy học phí hệ sau đại học (thạc sĩ, tiến sĩ)."""
    return get_hoc_phi("sau đại học")