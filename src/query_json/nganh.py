"""
nganh.py
Thông tin tĩnh về ngành: chỉ tiêu, tổ hợp, phương thức xét tuyển.

Nguồn dữ liệu:
  - chi_tieu_to_hop_2025.json
  - chi_tieu_tuyen_sinh_2026.json
  - to_hop_mon_thi.json
"""

from ._loader import load
from ._utils  import normalize, match_nganh, not_found, ok


# ── Chỉ tiêu & tổ hợp theo ngành ─────────────────────────────────────────────

def get_chi_tieu_nganh(ten_nganh: str) -> dict:
    """
    Lấy chỉ tiêu, tổ hợp và phương thức xét tuyển của ngành.

    Args:
        ten_nganh: Tên hoặc mã ngành. VD: "Công nghệ thông tin", "7480201"

    Returns:
        {
            "found": True,
            "ket_qua": [{
                "ma_nganh", "ten_nganh", "nhom",
                "chi_tieu", "to_hop", "phuong_thuc", "nam"
            }]
        }
    """
    data    = load("chi_tieu")
    results = [
        d for d in data
        if match_nganh(ten_nganh, d["ten_nganh"])
        or ten_nganh == d.get("ma_nganh", "")
    ]

    if not results:
        return not_found(f"Không tìm thấy ngành '{ten_nganh}'.")

    return ok(ket_qua=[
        {
            "ma_nganh"   : d["ma_nganh"],
            "ten_nganh"  : d["ten_nganh"],
            "nhom"       : d.get("nhom", ""),
            "chi_tieu"   : d["chi_tieu"],
            "to_hop"     : d["to_hop"],
            "phuong_thuc": d["phuong_thuc"],
            "nam"        : d.get("nam", 2025),
        }
        for d in results
    ])


def get_nganh_theo_to_hop(ma_to_hop: str) -> dict:
    """
    Tìm tất cả ngành xét tuyển bằng tổ hợp cho trước.

    Args:
        ma_to_hop: VD: "D01", "A00", "X25"

    Returns:
        {
            "found": True,
            "ma_to_hop": str,
            "mon_thi"  : [...],
            "so_nganh" : int,
            "nganh_list": [{"ma_nganh", "ten_nganh", "nhom", "chi_tieu"}]
        }
    """
    data    = load("chi_tieu")
    ma      = ma_to_hop.upper()
    results = [d for d in data if ma in [t.upper() for t in d["to_hop"]]]

    # Lấy môn thi của tổ hợp
    mon_thi = get_mon_thi_to_hop(ma).get("mon", [])

    if not results:
        return not_found(f"Không có ngành nào dùng tổ hợp {ma}.")

    return ok(
        ma_to_hop  = ma,
        mon_thi    = mon_thi,
        so_nganh   = len(results),
        nganh_list = [
            {
                "ma_nganh" : d["ma_nganh"],
                "ten_nganh": d["ten_nganh"],
                "nhom"     : d.get("nhom", ""),
                "chi_tieu" : d["chi_tieu"],
            }
            for d in sorted(results, key=lambda x: x["ten_nganh"])
        ],
    )


def get_chi_tieu_tong_2026() -> dict:
    """
    Lấy tổng chỉ tiêu và chỉ tiêu theo hệ đào tạo năm 2026.

    Returns:
        {"found": True, "truong", "nam", "tong_chi_tieu", "chi_tieu", "ghi_chu"}
    """
    raw = load("chi_tieu_2026")
    return ok(
        truong        = raw["truong"],
        nam           = raw["nam"],
        tong_chi_tieu = raw["tong_chi_tieu"],
        chi_tieu      = raw["chi_tieu"],
        ghi_chu       = raw.get("ghi_chu", ""),
    )


# ── Tổ hợp môn thi ───────────────────────────────────────────────────────────

def get_mon_thi_to_hop(ma_to_hop: str) -> dict:
    """
    Lấy danh sách môn thi của tổ hợp.

    Args:
        ma_to_hop: VD: "A00", "D01", "X25"

    Returns:
        {"found": True, "ma": str, "mon": ["Toán", "Vật lý", ...]}
    """
    data = load("to_hop")
    ma   = ma_to_hop.upper()
    for item in data:
        if item["ma"].upper() == ma:
            return ok(ma=item["ma"], mon=item["mon"])
    return not_found(f"Không tìm thấy tổ hợp '{ma_to_hop}'.")


def get_tat_ca_to_hop() -> list[dict]:
    """Trả về toàn bộ danh sách tổ hợp môn thi."""
    return load("to_hop")
