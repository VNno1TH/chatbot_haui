"""
diem_chuan.py
Tra cứu điểm chuẩn theo ngành, năm, phương thức.

Nguồn dữ liệu: diem_chuan_2023_2024_2025.json
Cấu trúc mỗi record:
  {
    "ma_nganh", "ten_nganh", "nhom_nganh",
    "nam", "phuong_thuc_code", "phuong_thuc_ten",
    "diem_chuan", "thang_diem",
    "cac_phuong_thuc_ap_dung"  (chỉ có ở năm 2025)
  }
"""

from ._loader import load
from ._utils  import match_nganh, not_found, ok

# Map tên hiển thị cho từng phương thức
PT_NAME = {
    "PT2"  : "Học bạ / Chứng chỉ quốc tế",
    "PT3"  : "Thi THPT",
    "PT4"  : "Đánh giá năng lực (HSA - ĐHQG HN)",
    "PT5"  : "Đánh giá tư duy (TSA - ĐHBK HN)",
    "PT6"  : "Đánh giá tư duy (TSA cũ)",
    "chung": "Chung (áp dụng cho nhiều phương thức)",
}


def get_diem_chuan(
    ten_nganh   : str,
    nam         : int | None = None,
    phuong_thuc : str | None = None,
) -> dict:
    """
    Lấy điểm chuẩn của ngành.

    Args:
        ten_nganh   : Tên hoặc mã ngành.
        nam         : Năm xét tuyển (2023/2024/2025). None = tất cả.
        phuong_thuc : Mã PT ("PT3", "PT5"...). None = tất cả.

    Returns:
        {
            "found"    : True,
            "ten_nganh": str,
            "ma_nganh" : str,
            "ket_qua"  : [{
                "nam", "phuong_thuc", "phuong_thuc_ten",
                "diem_chuan", "thang_diem",
                "ap_dung_cho" (optional)
            }]
        }
    """
    data = load("diem_chuan")

    # Lọc theo tên / mã ngành
    results = [
        d for d in data
        if match_nganh(ten_nganh, d["ten_nganh"])
        or ten_nganh == d.get("ma_nganh", "")
    ]
    if not results:
        return not_found(f"Không tìm thấy ngành '{ten_nganh}' trong dữ liệu điểm chuẩn.")

    # Lọc theo năm
    if nam is not None:
        results = [d for d in results if d["nam"] == nam]
        if not results:
            return not_found(
                f"Không có dữ liệu điểm chuẩn năm {nam} cho ngành '{ten_nganh}'."
            )

    # Lọc theo phương thức
    if phuong_thuc:
        pt = phuong_thuc.upper()
        filtered = [
            d for d in results
            if d["phuong_thuc_code"].upper() == pt
            or pt in [p.upper() for p in d.get("cac_phuong_thuc_ap_dung", [])]
        ]
        # Nếu không tìm thấy với PT cụ thể → trả TẤT CẢ (không filter)
        if not filtered:
            pass  # results vẫn là toàn bộ, không gán filtered
        else:
            results = filtered

    # Format kết quả
    ten = results[0]["ten_nganh"]
    ma  = results[0]["ma_nganh"]

    ket_qua = []
    for d in sorted(results, key=lambda x: (x["nam"], x["phuong_thuc_code"])):
        item = {
            "nam"            : d["nam"],
            "phuong_thuc"    : d["phuong_thuc_code"],
            "phuong_thuc_ten": PT_NAME.get(d["phuong_thuc_code"], d["phuong_thuc_ten"]),
            "diem_chuan"     : d["diem_chuan"],
            "thang_diem"     : d.get("thang_diem", 30),
        }
        if "cac_phuong_thuc_ap_dung" in d:
            item["ap_dung_cho"] = d["cac_phuong_thuc_ap_dung"]
        ket_qua.append(item)

    return ok(ten_nganh=ten, ma_nganh=ma, ket_qua=ket_qua)


def get_diem_chuan_moi_nhat(ten_nganh: str) -> dict:
    """
    Lấy điểm chuẩn năm gần nhất hiện có của ngành.

    Args:
        ten_nganh: Tên hoặc mã ngành.
    """
    data = load("diem_chuan")
    results = [
        d for d in data
        if match_nganh(ten_nganh, d["ten_nganh"])
        or ten_nganh == d.get("ma_nganh", "")
    ]
    if not results:
        return not_found(f"Không tìm thấy ngành '{ten_nganh}'.")

    nam_max = max(d["nam"] for d in results)
    return get_diem_chuan(ten_nganh, nam=nam_max)


def get_lich_su_diem_chuan(ten_nganh: str) -> dict:
    """
    Lấy toàn bộ lịch sử điểm chuẩn các năm của ngành.
    Tiện dùng để phân tích xu hướng tăng/giảm.
    """
    return get_diem_chuan(ten_nganh)


def get_diem_chuan_theo_khoa(
    ten_khoa    : str,
    nam         : int = 2025,
    phuong_thuc : str | None = None,
) -> dict:
    """
    Lấy điểm chuẩn TẤT CẢ ngành trong một khoa — phục vụ câu hỏi multi-hop.

    Ví dụ: "26 điểm có đậu ngành nào trong trường CNTT?"
    → traverse: khoa CNTT → [9 ngành] → điểm chuẩn từng ngành → so sánh

    Args:
        ten_khoa    : Tên khoa. VD: "CNTT", "Kinh tế"
        nam         : Năm xét tuyển (mặc định 2025)
        phuong_thuc : Lọc theo PT cụ thể. None = lấy tất cả

    Returns:
        {
            "found"     : True,
            "ten_khoa"  : str,
            "nam"       : int,
            "so_nganh"  : int,
            "ket_qua"   : [{
                "ma_nganh", "ten_nganh",
                "diem_chuan", "phuong_thuc", "thang_diem"
            }]
        }
    """
    from ._utils import normalize as _norm

    data = load("diem_chuan")

    # Chuẩn hóa tên khoa → nhom_nganh
    # Import tại đây để tránh circular import
    from .nganh import _resolve_khoa
    nhom = _resolve_khoa(ten_khoa)

    if nhom is None:
        # Thử tìm trực tiếp theo nhom_nganh trong diem_chuan JSON
        q    = _norm(ten_khoa)
        nhom_candidates = set(
            d["nhom_nganh"] for d in data
            if q in _norm(d.get("nhom_nganh", ""))
        )
        if not nhom_candidates:
            return not_found(
                f"Không nhận ra khoa '{ten_khoa}'. "
                f"Thử: CNTT, Cơ khí, Kinh tế, Du lịch, Dệt may, Ngôn ngữ, Thực phẩm."
            )
        nhom = next(iter(nhom_candidates))

    # Lọc theo khoa + năm
    results = [
        d for d in data
        if d.get("nhom_nganh") == nhom and d["nam"] == nam
    ]

    if not results:
        return not_found(
            f"Không có dữ liệu điểm chuẩn năm {nam} cho khoa '{ten_khoa}'."
        )

    # Lọc theo phương thức nếu có
    if phuong_thuc:
        pt = phuong_thuc.upper()
        filtered = [
            r for r in results
            if r["phuong_thuc_code"].upper() == pt
            or pt in [p.upper() for p in r.get("cac_phuong_thuc_ap_dung", [])]
        ]
        if filtered:
            results = filtered

    # Deduplicate: mỗi ngành chỉ lấy 1 record (ưu tiên PT3/chung)
    seen: dict[str, dict] = {}
    for r in sorted(results, key=lambda x: (
        0 if x["phuong_thuc_code"] in ("chung", "PT3") else 1,
        x["phuong_thuc_code"]
    )):
        ma = r["ma_nganh"]
        if ma not in seen:
            seen[ma] = r

    ket_qua = sorted(seen.values(), key=lambda x: x["diem_chuan"], reverse=True)

    return ok(
        ten_khoa  = ten_khoa,
        nhom_nganh= nhom,
        nam       = nam,
        so_nganh  = len(ket_qua),
        ket_qua   = [
            {
                "ma_nganh"   : r["ma_nganh"],
                "ten_nganh"  : r["ten_nganh"],
                "diem_chuan" : r["diem_chuan"],
                "phuong_thuc": r["phuong_thuc_code"],
                "thang_diem" : r.get("thang_diem", 30),
            }
            for r in ket_qua
        ],
    )